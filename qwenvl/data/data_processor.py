import json
import random
import logging
import re
import os
import csv
import time
import yaml
import itertools
from dataclasses import dataclass
from typing import Dict, Sequence, List, Any
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_pool = []
    video_pool = []

    # Build media pools with absolute paths
    if isinstance(item, dict) and 'video_path' in item:
        videos = [item['video_path'].replace('/mnt/yifanyang/', '/blob/')]
        video_pool = [{"type": "video", "video": vid} for vid in videos]
        caption = random.choice([item['caption'], item['short_caption']])
        item = {'conversations': [{'from': 'human', 'value': '<video>\nDescribe this video.'}, {'from': 'gpt', 'value': caption}]}
    elif "image" in item:
        base_path = item["image_dir"]
        images = item["image"]
        if isinstance(images, str):
            images = [images]
        image_pool = [{"type": "image", "image": os.path.join(base_path, img)} for img in images]
        
    # check if image_pool and video_pool are both empty
    if not image_pool and not video_pool:
        raise ValueError("No images or videos found in the data item.")
    
    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []

            # Ensure placeholders match available media
            if video_pool:
                text = text.replace("<image>", "<video>")
            elif image_pool:
                text = text.replace("<video>", "<image>")

            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # If any media remain unused, attach them to the first user turn to aviod droppping
    if image_pool or video_pool:
        for msg in messages:
            if msg["role"] == "user":
                msg["content"].extend(image_pool)
                msg["content"].extend(video_pool)
                image_pool = []
                video_pool = []
                break
        # If no user turn exists, just ignore the leftover without failing

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
    task_type="understanding",
) -> Dict:
    if isinstance(sources, (str, np.str_)):
        sources = [sources]
    if len(sources) != 1 :
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    messages = _build_messages(source)

    # Get input_ids and labels
    full_result = processor.apply_chat_template(
        messages, task=task_type, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def clean_caption(caption: str):
    caption = caption.strip()
    removed = False
    sentences = caption.split(". ")

    NEGATIVE_KEYWORDS = [
        "no", "not", "unreadable", "invisible", "illegible", "missing", "absent"
    ]
    TARGET_KEYWORDS = [
        "text", "caption", "subtitle", "word", "writing"
    ]

    if sentences:
        last = sentences[-1].rstrip(".").lower()
        if any(neg in last for neg in NEGATIVE_KEYWORDS) and any(
            tgt in last for tgt in TARGET_KEYWORDS
        ):
            sentences = sentences[:-1]
            removed = True

    cleaned_caption = ". ".join(s.rstrip(".") for s in sentences).strip()
    if cleaned_caption and not cleaned_caption.endswith("."):
        cleaned_caption += "."

    return cleaned_caption, removed


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            # self.get_rope_index = get_rope_index_3
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        '''
        # Load Video Data
        if data_args.add_video_data:
            pretrain_data_path = "/blob/dyb/processed_data/koala/video_captions_all.json"
            print(f"Loading from {pretrain_data_path} ...")
            with open(pretrain_data_path, 'r', encoding='utf-8') as f:
                list_data_dict = json.load(f)
            print(f"[OK] {pretrain_data_path} | entries: {len(list_data_dict)}")
            pretrain_data_path = "/blob/dyb/processed_data/IPOW_VIDU/test_videos_dataset.json"
            print(f"Loading from {pretrain_data_path} ...")
            with open(pretrain_data_path, 'r', encoding='utf-8') as f:
                list_data_dict.extend(json.load(f))
            print(f"[OK] {pretrain_data_path} | entries: {len(list_data_dict)}")
        '''

        # Load Video Data
        if data_args.add_video_data:
            data_dir = "/blob/dyb/processed_data"
            for root, dirs, files in os.walk(data_dir):
                dirs[:] = [d for d in dirs if d != 'videos']
                if 'video_captions_all_long_short.json' in files:
                    removed_count = 0
                    too_short_count = 0
                    json_path = os.path.join(root, 'video_captions_all_long_short.json')
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for item in data:
                        if "caption" in item and isinstance(item["caption"], str):
                            if len(item["caption"]) <= 10:
                                too_short_count += 1
                                continue
                            cleaned, removed = clean_caption(item["caption"])
                            item["caption"] = cleaned
                            if removed:
                                removed_count += 1

                    list_data_dict.extend(data)
                    print(
                        f"[OK] {json_path} | "
                        f"entries: {len(data)}, "
                        f"cleaned: {removed_count}, "
                        f"too short: {too_short_count}"
                    )

        # newly add
        pretrain_data_path = "/blob/dyb/processed_data/koala/video_captions_vbench_related.json"
        print(f"Loading from {pretrain_data_path} ...")
        with open(pretrain_data_path, 'r', encoding='utf-8') as f:
            list_data_dict.extend(json.load(f))
        print(f"[OK] {pretrain_data_path} | entries: {len(list_data_dict)}")
        pretrain_data_path = "/blob/dyb/processed_data/IPOW_VIDU/test_videos_dataset.json"
        print(f"Loading from {pretrain_data_path} ...")
        with open(pretrain_data_path, 'r', encoding='utf-8') as f:
            list_data_dict.extend(json.load(f))
        print(f"[OK] {pretrain_data_path} | entries: {len(list_data_dict)}")
        
        # Load Image Data
        if data_args.add_image_data:
            # image_json_dir = ['/blob/waq/playground/data/llava_1_6/llava_next_raw_format_processed.json', '/blob/waq/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json']
            # image_dirs = ['/blob/waq/playground/data/llava_1_6/images', '/blob/waq/playground/data/LLaVA-Pretrain/images']
            image_json_dir = ['/mnt/yifanyang/waq/playground/data/llava_1_6/llava_next_raw_format_processed.json']
            image_dirs = ['/mnt/yifanyang/waq/playground/data/llava_1_6/images']
            yaml_path = "/mnt/yifanyang/dyb/yijia_mid_stage.yaml"
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
            image_json_dir.extend([item["json_path"] for item in cfg["datasets"]])
            image_dirs.extend(["/mnt/yifanyang/hwq/data/llava_instruct/images" for _ in cfg["datasets"]])
            for image_dir, image_json in zip(image_dirs, image_json_dir):
                try:
                    with open(image_json.replace('/blob', '/mnt/yifanyang'), "r") as f:
                        annotations = json.load(f)
                except:
                    print(f"### failed to load image dataset {image_json}")
                    continue
                for ann in annotations:
                    ann['image_dir'] = image_dir
                print(f"### add image dataset {image_json}: {len(annotations)}")
                list_data_dict += annotations
            if data_args.show_data_structure:    
                print("Image Data Structure Example:", list_data_dict[-1])
            print("=" * 100)
        
        print(f"Total training samples: {len(list_data_dict)}")
        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_other_retries = 100

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass
        
        # try random samples
        for attempt_idx in range(num_other_retries):
            try:
                rand_index = random.randint(0, len(self.list_data_dict) - 1)

                sources = self.list_data_dict[rand_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample

            except Exception as e:
                print(f"[Try random #{attempt_idx}] Failed random_index={rand_index}. Error: {e}")

        # final fail
        print(f"[Final Fail] sample index={i} permanently failed. Skip this sample.")
        return None
            

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        # Randomly choose understanding or generation task
        p = 0.0
        if random.random() < p:
            task_type = "understanding"
        else:
            task_type = "generation"
        
        # Preprocess the data
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            task_type=task_type,
        )
        
        # Check again here
        if not "image_grid_thw" in data_dict and not "video_grid_thw" in data_dict:
            raise ValueError("Check again! No images or videos found in the data item.")
        
        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        data_dict["task_type"] = task_type

        # Test decoding
        '''
        text = self.processor.tokenizer.decode(data_dict["input_ids"][0], skip_special_tokens=False)
        labels = data_dict["labels"][0]
        labels = [tid if tid != -100 else self.processor.tokenizer.pad_token_id for tid in labels]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)
        '''

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        batch["task_type"] = instances[0]["task_type"]
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
