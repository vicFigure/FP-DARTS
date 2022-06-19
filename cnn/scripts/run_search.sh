#!/bin/bash
LOG_DIR=logs
NUM_COMPARE=5
CONFIG="--no_restrict"
saltype='sgas_1'

BATCHSIZE=64

let j=0
#for i in $(seq 0 3);do
for i in 0 1;do
    let SLURM_ARRAY_TASK_ID=$i
    echo $BASE_DIR $j
    python train_search.py --gpu $j $CONFIG --batch_size $BATCHSIZE --sal_type $saltype --num_compare $NUM_COMPARE --task $i > $LOG_DIR/$saltype-compare$NUM_COMPARE$CONFIG-task$i.log  2>&1 &
    let j=$j+1
done

