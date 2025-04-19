#!/bin/bash
deepspeed --master_port 9826 --include="localhost:0,1,2,3,4,5,6,7"\
    tabot/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ckpt/GroundingGPT-7B \
    --data_path data/TAU/annotations/train_all.json \
    --video_folder data/TAU/ \
    --image_folder data/TAU/ \
    --tune_mm_mlp_adapter False \
    --vision_tower ckpt/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True\
    --image_aspect_ratio square_nocrop\
    --bf16 True\
    --output_dir ./workdirs/tabot \
    --num_train_epochs 1\
    --per_device_train_batch_size 24\
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 1\
    --evaluation_strategy "no"\
    --save_strategy "steps"\
    --save_steps 500\
    --save_total_limit 1\
    --learning_rate 2e-5\
    --weight_decay 0.\
    --warmup_ratio 0.03\
    --lr_scheduler_type cosine\
    --logging_steps 10\
    --model_max_length 2048\
    --gradient_checkpointing True\
    --lazy_preprocess True\
    --report_to wandb


