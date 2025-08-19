#!/bin/bash
MODELS=50

NAME="GCB-Clone-Detection"
TASK_PATH="GraphCodeBERT/Clone-Detection/Morph"

# Trap sigint, stop docker and break the loop
function cleanup {
    echo "Stopping Docker containers..."
    docker stop $(docker ps -q)
    exit 0
}
trap cleanup SIGTERM SIGINT

for ((i = 0; i < MODELS; i++)); do
    echo "Running model $i"
    CMD="docker run --gpus all --rm -a stdout -v $(pwd):/root/green green_env /bin/bash -c 'cd /root/green/${TASK_PATH}; python3 train_pareto_front.py --model $i'"
    eval $CMD
done