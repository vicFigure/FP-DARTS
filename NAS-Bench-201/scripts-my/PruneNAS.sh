#!/bin/bash
# bash ./scripts-search/algos/GDAS.sh cifar10 0 -1
echo script name: $0
TORCH_HOME="path/to/NAS-BENCH-201/benchmark/file"
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi
gpu=( $@ )
gpu_num=$#
echo $gpu_num

dataset=cifar10
BN=1
seed=-1
channel=16
num_cells=5
max_nodes=4
space=nas-bench-201

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="${TORCH_HOME}/pytorch/NAS_PruneNAS/data"
else
  data_path="${TORCH_HOME}/pytorch/NAS_PruneNAS/data"
fi
#benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth
benchmark_file=${TORCH_HOME}/data_model/NAS-Bench-201-v1_1-096897.pth

save_dir=./output/search-cell-${space}/ROME_HPO-${dataset}-BN${BN}
LOG_DIR=logs

let j=0
for i in $(seq 3 3);do
#for i in 2 3;do
  echo $j ${gpu[$j]}
  CUDA_VISIBLE_DEVICES=${gpu[$j]} OMP_NUM_THREADS=4 python -u ./exps/algos/PruneNAS.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--arch_nas_dataset ${benchmark_file} \
	--config_path configs/nas-benchmark/algos/PruneNAS.config \
	--sal_type 'task' --num_compare 5 --reg_alpha 0.1\
    --track_running_stats ${BN} \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed} --task $i > $LOG_DIR/1-task$i.log  2>&1 &
  let j=($j+1)%$gpu_num
done
