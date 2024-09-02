#!/bin/bash
#
export WANDB_MODE=disabled
./train.py --output_dir 'models/save' \
           --eval_strategy 'epoch' \
           --num_train_epochs 3  \
           --logging_steps 100 \
           --per_device_train_batch_size 1 \
           --per_device_eval_batch_size 1

