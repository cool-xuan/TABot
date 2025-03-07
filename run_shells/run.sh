SHELL_NAMES=(
    "cc_og_ag_tl_wRef_wOB_e3_addTaskFlag_onLEGO"
)

# bash run_shells/all_gpu_detect.sh

for SHELL_NAME in ${SHELL_NAMES[@]}
do 
    bash run_shells/gpu_detect.sh 0
    bash run_shells/train/${SHELL_NAME}.sh
    bash run_shells/gpu_detect.sh 0
    bash run_shells/eval/${SHELL_NAME}.sh
done
