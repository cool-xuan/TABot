
#!/bin/bash

LEGO_MODEL="./ckpt/GroundingGPT-7B"
FT_MODEL="./workdirs/0625/cc_og_ag_tl_wRef_wOB_e3_addTaskFlag_onLEGO"
MODAL="imageVideo"
DATE="0625"
CKPT_NAME="cc_og_ag_tl_wRef_wOB_e3_addTaskFlag_onLEGO"

export WANDB_MODE=offline
deepspeed --master_port 9826 --include="localhost:0,1,2,3,4,5,6,7"\
    lego/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${LEGO_MODEL} \
    --data_path data/AllAccidentDataset/json_files_trainValTest/cc_og_qg_tl_conversations_all_train.json \
    --video_folder data/AllAccidentDataset/\
    --image_folder data/AllAccidentDataset/\
    --tune_mm_mlp_adapter False\
    --vision_tower ./ckpt/clip-vit-large-patch14-336\
    --mm_vision_select_layer -2\
    --mm_use_im_start_end True\
    --image_aspect_ratio square_nocrop\
    --bf16 True\
    --fp16 False\
    --output_dir ./workdirs/${DATE}/${CKPT_NAME} \
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
    --logging_steps 1\
    --tf32 True\
    --model_max_length 2048\
    --gradient_checkpointing True\
    --lazy_preprocess True\
    --report_to wandb


