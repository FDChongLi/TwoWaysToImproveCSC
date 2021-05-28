#!/usr/bin/env bash

job_name=adv_train_14_01_5
total_epochs=2
init_model=model
generate_clean=True
train_adv_threshold=0.01
test_adv_threshold=0.02
gpu=0
data_path=../data/sighan14_test
# data_path=../data/sighan13
adv_lr=3e-6
finetune_lr=6e-6

timestamp=`date "+%Y-%m-%d-%H-%M-%S"`
max_seq_length=180
do_lower_case=true
graph_dir="../data/gcn_graph.ty_xj/"
batch_size=8
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

test_model init_test ${job_name} ${bert_path} ${init_model} ${data_path} ${gpu}
adversarial_attack init_attack ${job_name} ${bert_path} ${init_model} ${data_path} ${generate_clean} False ${test_adv_threshold} ${gpu}

model_dir=${init_model}
adv_samples_dir=log_${job_name}/samples

if [ ! -d "${adv_samples_dir}" ]; then
mkdir -p ${adv_samples_dir}
cp ${data_path}/TestInput.txt ${adv_samples_dir}
cp ${data_path}/TestTruth.txt ${adv_samples_dir}
fi

for epoch in $(seq 1 ${total_epochs})
do

echo "***${job_name},epoch-${epoch} start!***"

# generate the adversarial samples for train
# Usage adversarial_attack <TASKNAME> <JOB_NAME> <BERT_PATH> <MODEL_PATH> <TEST_PATH> <CLEAN> <ATTACK_TRAIN> <THRESHOLD> <GPU>'
adversarial_attack attack_${epoch} ${job_name} ${bert_path} ${model_dir} ${data_path} ${generate_clean} True ${train_adv_threshold} ${gpu}
cp -f log_${job_name}/attack_${epoch}_0/advTotalInput.txt ${adv_samples_dir}/TrainingInputAll.txt
cp -f log_${job_name}/attack_${epoch}_0/advTotalTruth.txt ${adv_samples_dir}/TrainingTruthAll.txt

# train 1 epoch with adversarial samples 
# Usage finetune <TASK_NAME> <JOB_NAME> <BERT_PATH> <MODEL_PATH> <FINETUNE_PATH> <EPOCHS> <LEARNING_RATE> <GPU>'
finetune finetune_adv_${epoch} ${job_name} ${bert_path} ${model_dir} ${adv_samples_dir} 1 ${adv_lr} ${gpu}
model_dir=log_${job_name}/finetune_adv_${epoch}_0

# train 2 epoch with original data 
finetune finetune_clean_${epoch} ${job_name} ${bert_path} ${model_dir} ${data_path} 2 ${finetune_lr} ${gpu}
model_dir=log_${job_name}/finetune_clean_${epoch}_0

# check the performance and robustness
test_model test_${epoch} ${job_name} ${bert_path} ${model_dir} ${data_path} ${gpu}
adversarial_attack test_adv_gen_${epoch} ${job_name} ${bert_path} ${model_dir} ${data_path} ${generate_clean} False ${test_adv_threshold} ${gpu}
cp -f log_${job_name}/test_adv_gen_${epoch}_0/advTotalInput.txt ${adv_samples_dir}/TestInput.txt
cp -f log_${job_name}/test_adv_gen_${epoch}_0/advTotalTruth.txt ${adv_samples_dir}/TestTruth.txt
test_model test_adv_${epoch} ${job_name} ${bert_path} ${model_dir} ${adv_samples_dir} ${gpu}
wait
done