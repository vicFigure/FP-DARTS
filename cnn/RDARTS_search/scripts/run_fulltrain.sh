#!/bin/sh
# RDARTS Evaluation
DATASET='svhn'
CONFIG="--layers 20 --init_channels 36"
#CONFIG=""
ID=20210821-111239
SEARCH_DIR=ckpt/search-RDARTS-$ID-task
LOG_DIR=test_logs_sdarts

BATCHSIZE=64

let j=0
#for i in $(seq 4 7);do
for i in 4;do
    let SLURM_ARRAY_TASK_ID=$i
    BASE_DIR="$SEARCH_DIR$i"
    echo $BASE_DIR $j
    python train.py --gpu $j $CONFIG --batch_size $BATCHSIZE --cutout --auxiliary --dataset $DATASET --base_dir $BASE_DIR > $LOG_DIR/$ID-task$i.log  2>&1 &
    let j=$j+1
done


