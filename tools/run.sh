
#!/bin/bash

CONFIG=projects/configs/vidar_pretrain/nusc_1_8_subset/traj_occ_nusc_2history_2future_v2.py
GPUS=8
PORT=${PORT:-35789}


# 函数：获取 GPU 0 的显存占用
get_gpu_memory_usage() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0
}

# 函数：运行 Python 代码
run_python_code() {
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic --no-validate
}

# 初始化时间戳
start_time=$(date +%s)

# 初始化标志变量，用于记录是否已经小于10G超过2分钟
flag=false

while true; do
    # 获取 GPU 0 的显存占用
    memory_used=$(get_gpu_memory_usage)

    # 打印显存占用情况
    echo "GPU 0 显存占用：$memory_used MB"

    # 如果显存占用小于10G，则记录时间戳
    if (( $memory_used < 10000 )); then
        # 如果标志变量为 false，设置标志变量为 true 并记录时间戳
        if [ "$flag" = false ]; then
            start_time=$(date +%s)
            flag=true
        fi
    else
        # 如果显存占用不小于10G，重置标志变量为 false
        flag=false
    fi

    # 检查标志变量是否为 true 并且持续时间是否超过2分钟
    if [ "$flag" = true ]; then
        current_time=$(date +%s)
        time_diff=$((current_time - start_time))
        if (( $time_diff > 200 )); then
            run_python_code
            # 重置标志变量和时间戳
            flag=false
            start_time=$(date +%s)
        fi
    fi

    # 等待50s后再次读取
    sleep 50
done