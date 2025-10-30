#!/bin/bash
# Usage: bash train_1gpu.sh /path/to/config.json

CONFIG=$1
MODEL_NAME=$(basename "$(dirname $CONFIG)")

PORT=11001   # 다른 학습 프로세스랑 겹치지 않게 임의 포트

torchrun --nproc_per_node=1 \
         --master_port=$PORT \
    train.py -c $CONFIG -m $MODEL_NAME
