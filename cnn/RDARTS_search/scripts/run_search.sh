#!/bin/bash
SLURM_ARRAY_JOB_ID=0
SPACE='s4'
DATASET='cifar100'
LOGDIR=logs_rdarts
EPOCHS=50
gpu=0

WARMUP=20
COMPARE=5

for i in $(seq 4 7);do
    let SLURM_ARRAY_TASK_ID=$i
    echo $i $gpu
    # don't compute hessian
    python train_search.py --save RDARTS --space $SPACE --dataset $DATASET --epochs $EPOCHS --gpu $gpu --seed -1 --task $SLURM_ARRAY_TASK_ID --warmup $WARMUP --sal_type task --num_compare $COMPARE > $LOGDIR/$DATASET-$SPACE-$SLURM_ARRAY_TASK_ID.log 2>&1 &
    let gpu=($gpu+1)%4
done

