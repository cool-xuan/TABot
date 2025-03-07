#!/bin/bash
# 设置监控的GPU ID
GPU_ID=${1:-0}
# 设置显存使用空闲阈值（单位MiB）
FREE_THRESHOLD=1000
# 循环直到满足条件
while true; do
    # 获取GPU显存使用情况
    MEMORY_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | awk '{print $1}')
    
    # 检查显存是否低于阈值
    if [[ "$MEMORY_USAGE" -le "$FREE_THRESHOLD" ]]; then
        echo "GPU $GPU_ID 现在是空闲的。"
        
        # 在这里执行你的训练任务，比如：
        # python train_your_model.py
        
        break # 退出循环
    else
        # 显存使用超过阈值，等待一段时间后再次检查
        echo "GPU $GPU_ID 正忙，当前显存使用量：$MEMORY_USAGE MB。等待60秒后再次检查..."
        sleep 60
    fi
done
# 执行训练任务
echo "开始训练任务..."
# 在这里插入你的训练命令
# 例如：python train.py