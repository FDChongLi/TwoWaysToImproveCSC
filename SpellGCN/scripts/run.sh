#!/usr/bin/env bash
job_name=assure_27w
task_name=15

# mode: 1-test;2-attack;3-finetune
is_test=1
is_attack=2
is_finetune=3

mode=${is_test}
# data_path=./data
data_path=../data/sighan15
gpu=3

# args for attack
test_adv_threshold=0.02
generate_clean=True
attack_trainOrNot=False

# args for finetune
finetune_epochs=30
finetune_lr=6e-6

init_model=../models/27w/15
temp_data_dir=data
batch_size=8
timestamp=`date "+%Y-%m-%d-%H-%M-%S"`
max_seq_length=180
do_lower_case=true
graph_dir="../data/gcn_graph.ty_xj/"
bert_path=/home/zhcy/TAAD/chinese_L-12_H-768_A-12

# Usage adversarial_attack <TASKNAME> <JOB_NAME> <BERT_PATH> <MODEL_PATH> <TEST_PATH> <CLEAN> <ATTACK_TRAIN> <THRESHOLD> <GPU>'
function adversarial_attack(){
local task_name=$1
local model_dir=$4
local attack_clean=$6
local attack_train=$7
local adv_threshold=$8
local gpu=$9
local lr=5e-5
local num_epochs=1

# Just for Attack
for i in $(seq 0 0)
do

output_dir=log_${2}/${task_name}_$i
log_dir=log_${2}/${task_name}_$i

if [ ! -d "${output_dir}" ]; then
mkdir -p ${output_dir}
fi

echo "Start running ${task_name} attack-task_${i} log to ${output_dir}.log"

CUDA_VISIBLE_DEVICES=$gpu python ../adv_spellgcn.py \
--job_name=$2 \
--task_name=${task_name} \
--do_train=False \
--do_eval=False \
--do_predict=True \
--attack_clean=${attack_clean} \
--attack_train=${attack_train} \
--data_dir=$5 \
--vocab_file=$3/vocab.txt \
--bert_config_file=$3/bert_config.json \
--max_seq_length=${max_seq_length} \
--max_predictions_per_seq=${max_seq_length} \
--train_batch_size=${batch_size} \
--eval_batch_size=${batch_size} \
--learning_rate=${lr} \
--adv_threshold=${adv_threshold} \
--num_train_epochs=${num_epochs} \
--keep_checkpoint_max=10 \
--random_seed=${i}000 \
--init_checkpoint=${model_dir} \
--graph_dir=${graph_dir} \
--output_dir=${output_dir} >> ${log_dir}.log 2>&1 &
done
wait
}

# Usage finetune <TASK_NAME> <JOB_NAME> <BERT_PATH> <MODEL_PATH> <FINETUNE_PATH> <EPOCHS> <LEARNING_RATE> <GPU>'
function finetune(){
local task_name=$1
local lr=$7
local num_epochs=$6
local gpu=$8
local model_dir=$4
local keep_checkpoint_max=$6

# TRAIN
for i in $(seq 0 0)
do

local output_dir=log_${2}/${task_name}_$i
local log_dir=log_${2}/${task_name}_$i

# ! -d "dir" : if dir does not exist
if [ ! -d "${output_dir}/src" ]; then
mkdir -p ${output_dir}/src
cp $0 ${output_dir}/src
cp ../*py ${output_dir}/src
fi

#sleep $i
echo "Start running ${task_name} task-${i} log to ${output_dir}.log"
CUDA_VISIBLE_DEVICES=$gpu python ../finetune_spellgcn.py \
--job_name=$2 \
--task_name=${task_name} \
--do_train=True \
--do_eval=False \
--do_predict=False \
--data_dir=$5 \
--vocab_file=$3/vocab.txt \
--bert_config_file=$3/bert_config.json \
--max_seq_length=${max_seq_length} \
--max_predictions_per_seq=${max_seq_length} \
--train_batch_size=${batch_size} \
--learning_rate=${lr} \
--num_train_epochs=${num_epochs} \
--keep_checkpoint_max=${keep_checkpoint_max} \
--random_seed=${i}000 \
--init_checkpoint=${model_dir} \
--graph_dir=${graph_dir} \
--output_dir=${output_dir} > ${log_dir}.log 2>&1 &
done
wait
}

# Usage test_model <TASK_NAME> <JOB_NAME> <BERT_PATH> <MODEL_PATH> <TEST_PATH> <GPU>
function test_model(){
local task_name=$1
local lr=5e-5
local num_epochs=10
local gpu=$6
local model_dir=$4

# Just for TEST
for i in $(seq 0 0)
do

output_dir=log_${2}/${task_name}_$i
log_dir=log_${2}/${task_name}_$i

if [ ! -d "${output_dir}/src" ]; then
mkdir -p ${output_dir}/src
cp $0 ${output_dir}/src
cp ../*py ${output_dir}/src
fi

if [ ! -d "${log_dir}" ]; then
mkdir -p ${log_dir}
fi

echo "Start running ${task_name} task-${i} log to ${output_dir}.log"
CUDA_VISIBLE_DEVICES=$gpu python ../run_spellgcn.py \
--job_name=$2 \
--task_name=${task_name} \
--do_train=False \
--do_eval=False \
--do_predict=True \
--data_dir=$5 \
--vocab_file=$3/vocab.txt \
--bert_config_file=$3/bert_config.json \
--max_seq_length=${max_seq_length} \
--max_predictions_per_seq=${max_seq_length} \
--train_batch_size=${batch_size} \
--learning_rate=${lr} \
--num_train_epochs=${num_epochs} \
--keep_checkpoint_max=10 \
--random_seed=${i}000 \
--init_checkpoint=${model_dir} \
--graph_dir=${graph_dir} \
--output_dir=${output_dir} >> ${log_dir}.log 2>&1 &
done
wait
}

echo "***${job_name}***"

if [ ${mode} -eq ${is_test} ]; then
echo "mode:test"
# Usage test_model <TASK_NAME> <JOB_NAME> <BERT_PATH> <MODEL_PATH> <TEST_PATH> <GPU>
test_model ${task_name} ${job_name} ${bert_path} ${init_model} ${data_path} ${gpu}
elif [ ${mode} -eq ${is_finetune} ]; then
echo "mode:finetune"
# Usage finetune <TASK_NAME> <JOB_NAME> <BERT_PATH> <MODEL_PATH> <FINETUNE_PATH> <EPOCHS> <LEARNING_RATE> <GPU>'
finetune ${task_name} ${job_name} ${bert_path} ${init_model} ${data_path} ${finetune_epochs} ${finetune_lr} ${gpu}
else
echo "mode:attack"
# Usage adversarial_attack <TASKNAME> <JOB_NAME> <BERT_PATH> <MODEL_PATH> <TEST_PATH> <CLEAN> <ATTACK_TRAIN> <THRESHOLD> <GPU>'
adversarial_attack ${task_name} ${job_name} ${bert_path} ${init_model} ${data_path} ${generate_clean} ${attack_trainOrNot} ${test_adv_threshold} ${gpu}
cp -f log_${job_name}/${task_name}_0/advTotalInput.txt ${temp_data_dir}/TestInput.txt
cp -f log_${job_name}/${task_name}_0/advTotalTruth.txt ${temp_data_dir}/TestTruth.txt
test_model ${task_name}_test ${job_name} ${bert_path} ${init_model} ${temp_data_dir} ${gpu}
fi

