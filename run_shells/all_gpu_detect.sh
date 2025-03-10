#!/bin/bash
# 设置显存使用空闲阈值（单位MiB）
FREE_THRESHOLD=1000
# GPU总数
GPU_COUNT=8

# 连续空闲轮数计数器
idle_rounds=0

# 循环直到满足条件，即两次全部GPU都处于空闲
while true; do
    # 当前轮空闲GPU计数
    idle_gpus=0
    
    for GPU_ID in $(seq 0 $((GPU_COUNT-1))); do
        # 获取GPU显存使用情况
        MEMORY_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | awk '{print $1}')

        # 检查显存是否低于阈值
        if [[ "$MEMORY_USAGE" -le "$FREE_THRESHOLD" ]]; then
            echo "GPU $GPU_ID 现在是空闲的。"
            # 增加空闲GPU的计数器
            ((idle_gpus++))
        else
            echo "GPU $GPU_ID 正忙，当前显存使用量：$MEMORY_USAGE MB。"
        fi
        
        # 检测间隔，每检测一张卡等60秒再检测下一张卡
        sleep 60
    done
    
    # 检查这一轮是否所有GPU都空闲
    if [[ "$idle_gpus" -eq "$GPU_COUNT" ]]; then
        ((idle_rounds++))
        echo "全部GPU在这一轮都空闲，已空闲连续 $idle_rounds 次。"
    else
        idle_rounds=0
        echo "这一轮不是所有GPU都空闲。重置空闲轮数计数器。"
    fi
    
    # 检查是否已经连续两轮所有GPU都空闲
    if [[ "$idle_rounds" -eq 2 ]]; then
        echo "检测到连续两次所有GPU都空闲，准备开始执行任务..."
        break
    fi
done

# 执行训练任务
echo "开始训练任务..."
# 在这里插入你的训练命令，例如：
# python train.py
