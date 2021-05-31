#!/usr/bin/env bash
task_name=sighan13

is_test=1 #test,load
is_attack=2 #test,load bft_adtrain.py
is_baseline=3 #train,valid,test no load
is_finetune=4 #train,valid,test,load
is_ad_train=5 #train,valid,test bft_adtrain.py
is_pretrain=6 #train no valid,test,load

mode=${is_pretrain}

save_dir='../save'
log_dir='../log'
gpu=0,1
gpu_num=2
seed=0
epoch=2
batch_size=20
lr=2e-5

train_data_path=''
valid_data_path=''
test_data_path=''
model_path='' #load path

train_ratio=0.02
attack_ratio=0.02

function test_model(){
  local task_name=$1
  local save_dir=$2
  local gpu=$3
  local gpu_num=$4
  local model_path=$5
  local test_data=$6
  local batch_size=$7
  local lr=$8
  local seed=$9
  local log_dir=${10}

for i in $(seq 0 0)
do

model_save_dir=${save_dir}/${task_name}
if [ ! -d "${model_save_dir}" ]; then
mkdir -p ${model_save_dir}
fi

CUDA_VISIBLE_DEVICES=$gpu python -u ../smb_train.py \
--task_name=${task_name} \
--gpu_num=${gpu_num} \
--load_model=True \
--load_path=${model_path} \
--do_test=True \
--test_data=${test_data} \
--batch_size=${batch_size} \
--learning_rate=${lr} \
--seed=${seed} >> ${log_dir}/${task_name}.log 2>&1 &
# --seed=${seed} 2>&1 &
done
wait
}

function attack_model(){
  local task_name=$1
  local save_dir=$2
  local gpu=$3
  local gpu_num=$4
  local model_path=$5
  local test_data=$6
  local batch_size=$7
  local lr=$8
  local seed=$9
  local log_dir=${10}
  local attack_ratio=${11}

for i in $(seq 0 0)
do

model_save_dir=${save_dir}/${task_name}
if [ ! -d "${model_save_dir}" ]; then
mkdir -p ${model_save_dir}
fi

CUDA_VISIBLE_DEVICES=$gpu python -u ../smb_adtrain.py \
--task_name=${task_name} \
--gpu_num=${gpu_num} \
--load_model=True \
--load_path=${model_path} \
--do_test=True \
--test_data=${test_data} \
--batch_size=${batch_size} \
--learning_rate=${lr} \
--attack_ratio=${attack_ratio} \
--seed=${seed} >> ${log_dir}/${task_name}.log 2>&1 &
# --seed=${seed} 2>&1 &
done
wait
}

function baseline_model(){
  local task_name=$1
  local save_dir=$2
  local gpu=$3
  local gpu_num=$4
  local train_data=$5
  local valid_data=$6
  local epoch=$7
  local batch_size=$8
  local lr=$9
  local seed=${10}
  local log_dir=${11}

for i in $(seq 0 0)
do

model_save_dir=${save_dir}/${task_name}
if [ ! -d "${model_save_dir}" ]; then
mkdir -p ${model_save_dir}
fi

CUDA_VISIBLE_DEVICES=$gpu python -u ../smb_train.py \
--task_name=${task_name} \
--gpu_num=${gpu_num} \
--load_model=False \
--do_train=True \
--train_data=${train_data} \
--do_valid=True \
--valid_data=${valid_data} \
--epoch=${epoch} \
--batch_size=${batch_size} \
--learning_rate=${lr} \
--do_save=True \
--save_dir=${model_save_dir} \
--seed=${seed} >> ${log_dir}/${task_name}.log 2>&1 &
# --seed=${seed} 2>&1 &
done
wait
}

function finetune_model(){
  local task_name=$1
  local save_dir=$2
  local gpu=$3
  local gpu_num=$4
  local model_path=$5
  local train_data=$6
  local valid_data=$7
  local epoch=$8
  local batch_size=$9
  local lr=${10}
  local seed=${11}
  local log_dir=${12}

for i in $(seq 0 0)
do

model_save_dir=${save_dir}/${task_name}
if [ ! -d "${model_save_dir}" ]; then
mkdir -p ${model_save_dir}
fi

CUDA_VISIBLE_DEVICES=$gpu python -u ../smb_train.py \
--task_name=${task_name} \
--gpu_num=${gpu_num} \
--load_model=True \
--load_path=${model_path} \
--do_train=True \
--train_data=${train_data} \
--do_valid=True \
--valid_data=${valid_data} \
--epoch=${epoch} \
--batch_size=${batch_size} \
--learning_rate=${lr} \
--do_save=True \
--save_dir=${model_save_dir} \
--seed=${seed} >> ${log_dir}/${task_name}.log 2>&1 &
# --seed=${seed} 2>&1 &
done
wait
}

function adtrain_model(){
  local task_name=$1
  local save_dir=$2
  local gpu=$3
  local gpu_num=$4
  local model_path=$5
  local train_data=$6
  local valid_data=$7
  local epoch=$8
  local batch_size=$9
  local lr=${10}
  local seed=${11}
  local log_dir=${12}
  local train_ratio=${13}
  local attack_ratio=${14}

for i in $(seq 0 0)
do

model_save_dir=${save_dir}/${task_name}
if [ ! -d "${model_save_dir}" ]; then
mkdir -p ${model_save_dir}
fi

CUDA_VISIBLE_DEVICES=$gpu python -u ../smb_adtrain.py \
--task_name=${task_name} \
--gpu_num=${gpu_num} \
--load_model=True \
--load_path=${model_path} \
--do_train=True \
--train_data=${train_data} \
--do_valid=True \
--valid_data=${valid_data} \
--epoch=${epoch} \
--batch_size=${batch_size} \
--learning_rate=${lr} \
--do_save=True \
--save_dir=${model_save_dir} \
--train_ratio=${train_ratio} \
--attack_ratio=${attack_ratio} \
--seed=${seed} >> ${log_dir}/${task_name}.log 2>&1 &
# --seed=${seed} 2>&1 &
done
wait
}

function pretrain_model(){
  local task_name=$1
  local save_dir=$2
  local gpu=$3
  local gpu_num=$4
  local pretrain_corpus=$5
  local epoch=$6
  local batch_size=$7
  local lr=$8
  local seed=$9
  local log_dir=${10}

for i in $(seq 0 0)
do

model_save_dir=${save_dir}/${task_name}
if [ ! -d "${model_save_dir}" ]; then
mkdir -p ${model_save_dir}
fi

CUDA_VISIBLE_DEVICES=$gpu python -u ../smb_train.py \
--task_name=${task_name} \
--gpu_num=${gpu_num} \
--load_model=False \
--do_train=True \
--train_data=${pretrain_corpus} \
--epoch=${epoch} \
--batch_size=${batch_size} \
--learning_rate=${lr} \
--do_save=True \
--save_dir=${model_save_dir} \
--seed=${seed} >> ${log_dir}/${task_name}.log 2>&1 &
# --seed=${seed} 2>&1 &
done
wait
}

if [ ${mode} -eq ${is_test} ]; then
  echo "mode:test"
  test_model ${task_name} ${save_dir} ${gpu} ${gpu_num} ${model_path} ${test_data_path} ${batch_size} ${lr} ${seed} ${log_dir}
elif [ ${mode} -eq ${is_attack} ]; then
  echo "mode:attack"
  attack_model ${task_name} ${save_dir} ${gpu} ${gpu_num} ${model_path} ${test_data_path} ${batch_size} ${lr} ${seed} ${log_dir} ${attack_ratio}
elif [ ${mode} -eq ${is_baseline} ]; then
  echo "mode:baseline"
  baseline_model ${task_name} ${save_dir} ${gpu} ${gpu_num} ${train_data_path} ${valid_data_path} ${epoch} ${batch_size} ${lr} ${seed} ${log_dir}
elif [ ${mode} -eq ${is_finetune} ]; then
  echo "mode:finetune"
  finetune_model ${task_name} ${save_dir} ${gpu} ${gpu_num} ${model_path} ${train_data_path} ${valid_data_path} ${epoch} ${batch_size} ${lr} ${seed} ${log_dir}
elif [ ${mode} -eq ${is_ad_train} ]; then
  echo "mode:adtrain"
  adtrain_model ${task_name} ${save_dir} ${gpu} ${gpu_num} ${model_path} ${train_data_path} ${valid_data_path} ${epoch} ${batch_size} ${lr} ${seed} ${log_dir} ${train_ratio} ${attack_ratio}
elif [ ${mode} -eq ${is_pretrain} ]; then
  echo "mode:pretrain"
  pretrain_model ${task_name} ${save_dir} ${gpu} ${gpu_num} ${train_data_path} ${epoch} ${batch_size} ${lr} ${seed} ${log_dir}
fi
