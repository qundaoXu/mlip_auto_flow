#!/bin/bash
echo "success run bash command 1"

MAX_JOBS=${ALFLOW_MAX_JOBS:-30}  # 最大并行提交任务数（alflow.dft.max_jobs）
CURRENT_JOBS=0
TOTAL_JOBS=0
SUBMITTED_JOBS=0

# 统计总任务数
for dir in */; do
    if [ -f "${dir}std.run" ]; then
        ((TOTAL_JOBS++))
    fi
done

# 进度条函数
function progress_bar() {
    local current=$1
    local total=$2
    local width=50  # 进度条宽度
    local percent=$((current * 100 / total))
    local completed=$((current * width / total))
    local remaining=$((width - completed))
    
    printf "\r[%-*s] %d%% (%d/%d)" \
        "$width" \
        "$(printf '#%.0s' $(seq 1 $completed))$(printf ' %.0s' $(seq 1 $remaining))" \
        "$percent" \
        "$current" \
        "$total"
}

# 主循环
for dir in */; do
    if [ -f "${dir}std.run" ]; then
        # 等待空闲槽位
        while [ "$CURRENT_JOBS" -ge "$MAX_JOBS" ]; do
            sleep 2
            CURRENT_JOBS=$(squeue -u $USER | grep -v "JOBID" | wc -l)
            progress_bar $SUBMITTED_JOBS $TOTAL_JOBS
        done
        
        # 提交任务
        echo ""
        echo "Time:$(date) Submitting: ${dir}std.run"
        (cd "$dir" && sbatch std.run) &
        ((CURRENT_JOBS++))
        ((SUBMITTED_JOBS++))
        sleep 0.5
        
        # 更新进度条
        progress_bar $SUBMITTED_JOBS $TOTAL_JOBS
    fi
done

wait
echo ""
echo "All $SUBMITTED_JOBS jobs have been submitted"
echo ""
echo "Current job queue:"
squeue -u $USER
