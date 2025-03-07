DATE=$1
CKPT_NAME=$2
BASE_MODEL=$3
EVAL_TASKS=$4

# 初始化 GPU ID
GPU_ID=6

# 初始化声明一个名为tasks的关联数组
declare -A tasks

# 遍历任务列表
for task in $EVAL_TASKS; do
  if [[ "$task" == "og" || "$task" == "ag" || "$task" == "ccI" ]]; then
    # 模态设置为image
    modality="image"
  elif [[ "$task" == "tl" || "$task" == "ccV" ]]; then
    # 模态设置为video
    modality="video"
  else
    # 如果出现是未知任务类型，可以在这里处理异常情况
    echo "Unknown task: $task"
    continue
  fi
  # 根据任务和模态赋值，每个类型的test和val赋相同值
  tasks["$modality,$task,test","addTaskFlag"]=$GPU_ID
  GPU_ID=$(( (GPU_ID + 1) % 8 ))
  # tasks["$modality,$task,test","noAddTaskFlag"]=$GPU_ID
  # GPU_ID=$(( (GPU_ID + 1) % 8 ))
  tasks["$modality,$task,val","addTaskFlag"]=$GPU_ID
  GPU_ID=$(( (GPU_ID + 1) % 8 ))
  # tasks["$modality,$task,val","noAddTaskFlag"]=$GPU_ID
  # GPU_ID=$(( (GPU_ID + 1) % 8 ))
done

# 你的eval脚本名
EVAL_SCRIPT="run_shells/eval_single_task.sh"

# 为评估脚本添加执行权限
chmod +x "$EVAL_SCRIPT"

# 遍历tasks关联数组
for task_key in "${!tasks[@]}"; do
    read -r MODAL TASK KEY_WORDS SUFFIX <<<$(echo $task_key | tr ',' ' ')

    GPU=${tasks[$task_key]} # 从task_key获取GPU编号

    # 构建并运行命令
    (
        bash "$EVAL_SCRIPT" "$DATE" "$CKPT_NAME" "$BASE_MODEL" "$GPU" "$MODAL" "$TASK" "$KEY_WORDS" "$SUFFIX"
        echo "任务 $MODAL-$TASK-$KEY_WORDS-$SUFFIX 完成在 GPU $GPU"
    ) &
done

# 等待所有后台任务完成
wait
echo "所有任务已完成！"
