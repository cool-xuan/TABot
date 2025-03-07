#!/bin/bash

# 通用参数
DATE="0625"
CKPT_NAME="cc_og_ag_tl_wRef_wOB_e3_addTaskFlag_onLEGO"
BASE_MODEL="" # 如果BASE_MODEL始终为空，我们甚至不需要在命令行中包含它
EVAL_TASKS="og ag ccI tl ccV" # 任务设置

bash run_shells/mp_task_launcher.sh "${DATE}" "${CKPT_NAME}" "${BASE_MODEL}" "$EVAL_TASKS"
