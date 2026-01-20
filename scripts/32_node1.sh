#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
llm=Qwen/Qwen3-VL-2B-Instruct  # Using HuggingFace model ID
vq_path=/blob/dyb_output/icml2026/embedding_codebook_vanilla/checkpoint-55399/model.safetensors

# Training hyperparameters
lr=2e-5
batch_size=1
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=0_30_s_academic_v0_1,0_30_s_youtube_v0_1,0_30_s_activitynetqa,0_30_s_nextqa,0_30_s_perceptiontest,30_60_s_academic_v0_1,30_60_s_youtube_v0_1,30_60_s_activitynetqa,30_60_s_nextqa,30_60_s_perceptiontest,1_2_m_academic_v0_1,1_2_m_youtube_v0_1,1_2_m_activitynetqa,1_2_m_nextqa,2_3_m_academic_v0_1,2_3_m_youtube_v0_1,2_3_m_activitynetqa,2_3_m_nextqa
# datasets=0_30_s_academic_v0_1,0_30_s_youtube_v0_1,0_30_s_activitynetqa,0_30_s_nextqa,0_30_s_perceptiontest

# Output configuration
run_name="qwen3vl_2b_single_ema_codebook"
output_dir=/blob/dyb_output/icml2026/qwen3vl_2b_embedding_codebook_vanilla
export WANDB_PROJECT="icml_ar_ablation"

# Training arguments
args="
    --deepspeed ${deepspeed}
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --train_vq_wo_llm False \
    --add_image_data False \
    --add_video_data True \
    --vq_path ${vq_path} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_vqvae False \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 8388608 \
    --min_pixels 262144 \
    --video_max_pixels 33554432 \
    --video_min_pixels 1048576 \
    --video_max_frames 32 \
    --video_min_frames 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 100 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps 10 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nnodes=4 \
         --nproc_per_node=8 \
         --node_rank=1 \
         --master_addr=100.64.141.117 \
         --master_port=30000 \
         --max_restarts=10 \
         ${entry_file} ${args}


python /blob/thinking.py

