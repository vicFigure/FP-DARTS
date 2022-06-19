#!/bin/bash
# bash ./scripts-search/NAS-Bench-201/train-a-net.sh resnet 16 5
TORCH_HOME="path/to/NAS-BENCH-201/benchmark/file"
echo script name: $0
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

gpu=( $@ )
gpu_num=$#
echo $gpu_num

channel=16
num_cells=5
ID=20210117-093324
SEARCH_DIR=./output/search-cell-nas-bench-201/ROME_HPO-cifar10-BN1/$ID-task
LOG_DIR=test_logs
CONFIG="--default_hp"
#CONFIG=""


let j=0
for i in $(seq 0 1);do
  BASE_DIR="$SEARCH_DIR$i"
  echo $j ${gpu[$j]} $BASE_DIR
  CUDA_VISIBLE_DEVICES=${gpu[$j]} OMP_NUM_THREADS=4 python -u ./exps/NAS-Bench-201/main-my.py \
    $CONFIG --base_dir $BASE_DIR --save_dir $BASE_DIR --max_node 4 \
	--datasets cifar10 cifar10 cifar100 \
	--use_less 0 \
	--splits         1       0        0 \
	--xpaths $TORCH_HOME/data/ \
		 $TORCH_HOME/data/ \
		 $TORCH_HOME/data/ \
	--channel ${channel} --num_cells ${num_cells} \
	--workers 4 \
	--seeds 777 888 999 > $LOG_DIR/$ID$CONFIG-task$i.log  2>&1 &
  let j=($j+1)%$gpu_num
done
#	--datasets cifar10 cifar10 cifar100 ImageNet16-120 \
#	--splits         1       0        0              0 \
#		 $TORCH_HOME/data/ImageNet16 \
