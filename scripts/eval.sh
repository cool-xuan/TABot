#!/bin/bash
# Image task: Recognition and Description
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python tabot/evaluate/image_inference.py \
    --model_path workdirs/tabot \
    --json_path data/TAU/annotations/test/test_image_rd.json \
    --image_dir data/TAU \
    --output_dir outputs/tabot 

python evaluate/image_tasks_eval.py \
    --json_path outputs/tabot/test_image_rd.json 

# Image task: Spatial Grounding
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python tabot/evaluate/image_inference.py \
    --model_path workdirs/tabot \
    --json_path data/TAU/annotations/test/test_image_sg.json \
    --image_dir data/TAU \
    --output_dir outputs/tabot 

python evaluate/image_tasks_eval.py \
    --json_path outputs/tabot/test_image_sg.json

# Video task: Recognition and Description
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python tabot/evaluate/video_inference.py \
    --model_path workdirs/tabot \
    --json_path data/TAU/annotations/test/test_video_rd.json \
    --video_dir data/TAU \
    --output_dir outputs/tabot

python evaluate/video_tasks_eval.py \
    --json_path outputs/tabot/test_video_rd.json

# Video task: Temporal Localization
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python tabot/evaluate/video_inference.py \
    --model_path workdirs/tabot \
    --json_path data/TAU/annotations/test/test_video_tl.json \
    --video_dir data/TAU \
    --output_dir outputs/tabot

python evaluate/video_tasks_eval.py \
    --json_path outputs/tabot/test_video_tl.json