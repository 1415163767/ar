# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import random
import logging
import pathlib
import torch
import transformers
import sys
import numpy as np
from pathlib import Path
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    # Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer
import warnings

warnings.filterwarnings("ignore", message=".*torchvision.*video.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
local_rank = None

def seed_anything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set the seed for reproducibility
seed = 42
seed_anything(seed)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
        model.gen_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
        model.gen_head.requires_grad = False
    
    if model_args.tune_vqvae:
        for n, p in model.visual.vq.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.vq.named_parameters():
            p.requires_grad = False


def new_visual_forward(self, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, **kwargs):
    hidden_states = pixel_values if pixel_values is not None else pixel_values_videos
    hidden_states = hidden_states.to(dtype=self.patch_embed.proj.weight.dtype)
    hidden_states = self.patch_embed(hidden_states)
    is_image = pixel_values is not None

    grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for blk in self.blocks:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    hidden_states = self.merger(hidden_states)
    hidden_states, codebook_loss = self.vq(hidden_states, is_image=is_image)

    hidden_states = hidden_states + 0 * self.vq.dummy
    loss = codebook_loss["loss"] + 0 * self.vq.dummy

    return loss, hidden_states


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    print("=" * 100)

    if training_args.train_vq_wo_llm:
        import types
        from qwen3_vl import Qwen3VLVisionModel
        print("Training VQ-VAE without LLM")
        model = Qwen3VLVisionModel.from_pretrained('/mnt/data/dyb/pretrained_ckpts/qwen3vl_vision_only', cache_dir=training_args.cache_dir)
        # model = Qwen3VLVisionModel.from_pretrained('/zehui/dyb/pretrained_ckpts/qwen3vl_vision_only_32k', cache_dir=training_args.cache_dir)
        model.vq.dummy = torch.nn.Parameter(torch.zeros(1))
        model.forward = types.MethodType(new_visual_forward, model)
        data_args.model_type = "qwen3vl"
        data_args.use_tokenizer = False
        print(f'the initlized model is VQ the class is {model.__class__.__name__}')
    else:
        if "qwen3" in model_args.model_name_or_path.lower():
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            try:
                model.resize_token_embeddings(151936 + 1)   # <mask>
            except:
                model.resize_token_embeddings(151936 + 1, mean_resizing=False)
            if training_args.vq_path != "":
                from safetensors.torch import load_file
                visual = load_file(training_args.vq_path)
                model.visual.load_state_dict(visual, strict=False)
            data_args.model_type = "qwen3vl"
        elif "qwen2.5" in model_args.model_name_or_path.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            data_args.model_type = "qwen2.5vl"
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            data_args.model_type = "qwen2vl"

        print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')

        if data_args.data_flatten or data_args.data_packing:
            replace_qwen2_vl_attention_class()
        model.config.use_cache = False

        # Load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        data_args.use_tokenizer = True

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
    except:
        if "qwen3" in model_args.model_name_or_path.lower():
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
            )

    if training_args.gradient_checkpointing and not training_args.train_vq_wo_llm:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        if training_args.train_vq_wo_llm:
            for n, p in model.named_parameters():
                p.requires_grad = False
                if 'vq' in n:
                    print(f"Trainable Parameter: {n}")
                    p.requires_grad = True
            if torch.distributed.get_rank() == 0:
                print("=" * 100)
                trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
                print(f"Trainable Parameters: {trainable_parameters:.2f}M")
                print("=" * 100)
        else:
            set_model(model_args, model)
            if torch.distributed.get_rank() == 0:
                print("=" * 100)
                model.visual.print_trainable_parameters()
                print("=" * 100)
                model.model.print_trainable_parameters()
                print("=" * 100)
                all_parameters_visual = sum(p.numel() for p in model.model.visual.parameters()) / 1e6
                all_parameters_llm = sum(p.numel() for p in model.model.language_model.parameters()) / 1e9
                trainable_parameters_visual = sum(p.numel() for p in model.model.visual.parameters() if p.requires_grad) / 1e6
                trainable_parameters_llm = sum(p.numel() for p in model.model.language_model.parameters() if p.requires_grad) / 1e9
                print(f"All Parameters Visual: {all_parameters_visual:.2f}M, "f"Trainable Parameters Visual: {trainable_parameters_visual:.2f}M")
                print(f"All Parameters LLM: {all_parameters_llm:.2f}B, "f"Trainable Parameters LLM: {trainable_parameters_llm:.2f}B")
                print("=" * 100)

        '''
        if torch.sum(model.visual.vq.quantize.embedding.weight) == 0:
            print("Codebook is empty. Initializing...")
            codebook_tensor = torch.empty_like(model.visual.vq.quantize.embedding.weight)
            torch.nn.init.uniform_(codebook_tensor, -1.0 / 16384, 1.0 / 16384)
            model.visual.vq.quantize.embedding.weight.data.copy_(codebook_tensor)
        '''
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    print("=" * 100)
    if data_args.use_tokenizer:
        print("Using tokenizer in data module")
        trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )
    else:
        print("Not using tokenizer in data module")
        trainer = Trainer(
            model=model, processing_class=processor, args=training_args, **data_module
        )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

