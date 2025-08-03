#!/bin/bash
MODELS=20

NAME="GCB-Vulnerability-Detection"
TASK_PATH="GraphCodeBERT/Vulnerability-Detection/Morph"

for ((i = 0; i < MODELS; i++)); do
    echo "Running model $i"
    CMD="docker run --gpus all --rm -a stdout -v $(pwd):/root/green green_env /bin/bash -c 'cd /root/green/${TASK_PATH}; python3 many_objective.py --model-name model-$i.bin'"
done

for ((i = 0; i < MODELS; i++)); do
    echo "Running model $i"
    CMD="docker run --gpus all --rm -a stdout -v $(pwd):/root/green green_env /bin/bash -c 'cd /root/green/${TASK_PATH}; python3 many_objective.py --model-name model-$i.bin --use-flops'"
done