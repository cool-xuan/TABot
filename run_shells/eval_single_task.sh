#!/bin/bash

# 参数设置
DATE=$1
CKPT_NAME=$2
BASE_MODEL=${3:-""}
GPU=$4
MODAL=$5
TASK=$6
KEY_WORDS=$7
SUFFIX=$8

if [[ "${TASK}" == *"ag"* ]]; then
    JSON_DIR="data/AllAccidentDataset/json_files_trainValTest/tasks/accident_grounding"
    if [[ "${KEY_WORDS}" == *"val"* ]]; then
        JSON_FILE="ag_image_conversations_all_val.json"
    else
        JSON_FILE="ag_image_conversations_all_test.json"
    fi
elif [[ "${TASK}" == *"og"* ]]; then
    JSON_DIR="data/AllAccidentDataset/json_files_trainValTest/tasks/object_grounding"
    if [ "${KEY_WORDS}" == "val" ]; then
        JSON_FILE="og_image_conversations_all_val.json"
    else
        JSON_FILE="og_image_conversations_all_test.json"
    fi
elif [[ "${TASK}" == *"ccI"* ]]; then
    JSON_DIR="data/AllAccidentDataset/json_files_trainValTest/tasks/classfication_caption_image"
    if [ "${KEY_WORDS}" == "val" ]; then
        JSON_FILE="cc_image_conversations_all_val.json"
    else
        JSON_FILE="cc_image_conversations_all_test.json"
    fi
elif [[ "${TASK}" == *"ccV"* ]]; then
    JSON_DIR="data/AllAccidentDataset/json_files_trainValTest/tasks/classfication_caption_video"
    if [ "${KEY_WORDS}" == "val" ]; then
        JSON_FILE="cc_video_conversations_all_val.json"
    else
        JSON_FILE="cc_video_conversations_all_test.json"
    fi
else
    JSON_DIR="data/AllAccidentDataset/json_files_trainValTest/tasks/temporal_localization"
    if [ "${KEY_WORDS}" == "val" ]; then
        JSON_FILE="tl_video_conversations_all_val.json"
    else
        JSON_FILE="tl_video_conversations_all_test.json"
    fi
fi

if [ "${SUFFIX}" == "addTaskFlag" ]; then
    JSON_DIR="${JSON_DIR}/addTaskFlag"
fi

# 根据参数决定需不需要添加额外的flag
EXTRA_FLAGS=""
if [ "${SUFFIX}" != "" ]; then
    EXTRA_FLAGS="--shuffix ${SUFFIX}"
fi

# 检测模态，设定相应的文件路径和计算脚本
if [ "${MODAL}" == "image" ]; then
    CONVERT_SCRIPT="grounding_output_converter.py"
    EVAL_SCRIPT="grounding_output_bbox_eval.py"
else
    CONVERT_SCRIPT="temporal_output_converter.py"
    EVAL_SCRIPT="temporal_localization_eval.py"
fi

if [ "${BASE_MODEL}" == "" ]; then
    # 运行评估模型
    CUDA_VISIBLE_DEVICES=${GPU} \
    python lego/evaluate/${MODAL}Only-cli.py \
        --json_path ${JSON_DIR}/${JSON_FILE} \
        --${MODAL}_dir data/AllAccidentDataset \
        --output_dir outputs/${DATE} \
        --model_path workdirs/${DATE}/${CKPT_NAME} \
        --task ${TASK} \
        ${EXTRA_FLAGS}
else
    # 运行评估模型
    CUDA_VISIBLE_DEVICES=${GPU} \
    python lego/evaluate/${MODAL}Only-cli.py \
        --json_path ${JSON_DIR}/${JSON_FILE} \
        --${MODAL}_dir data/AllAccidentDataset \
        --output_dir outputs/${DATE} \
        --model_path workdirs/${DATE}/${CKPT_NAME} \
        --model_base ${BASE_MODEL} \
        --task ${TASK} \
        ${EXTRA_FLAGS}
fi

# 转换和评估输出
python outputs/metric_computer/${CONVERT_SCRIPT} --json_dir outputs/${DATE}/${CKPT_NAME}/${TASK} --key_words ${KEY_WORDS} ${EXTRA_FLAGS} --task ${TASK}
python outputs/metric_computer/${EVAL_SCRIPT} --json_dir outputs/${DATE}/${CKPT_NAME}/${TASK} --key_words ${KEY_WORDS} ${EXTRA_FLAGS} --task ${TASK}
