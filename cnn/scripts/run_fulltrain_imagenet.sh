#!/bin/sh
# RDARTS Evaluation
DATA_PATH=/data/wangxiaoxing/imagenet
CONFIG="--no_restrict"
#CONFIG=""
ID=20211211-143302
SEARCH_DIR=ckpt/search-EXP-$ID-task
LOG_DIR=test_logs_imagenet

BATCHSIZE=768

let j=0
#for i in $(seq 0 3);do
for i in 1;do
    let SLURM_ARRAY_TASK_ID=$i
    BASE_DIR=$SEARCH_DIR$i
    echo $BASE_DIR $j
    python train_imagenet.py --data $DATA_PATH --mode train $CONFIG --batch_size $BATCHSIZE --auxiliary --base_path $BASE_DIR --genotype_name 59 > $LOG_DIR/$ID-task$i.log 2>&1 &
    let j=$j+1
done

