#!/usr/bin/env bash

mkdir -p ./logs

seq_len=96
model_name=AHuber
root_path=./data/ETT
model_id_name=ETTm1
data_name=ETTm1
random_seed=2021

for pred_len in 96 192
do
    model_id="${model_id_name}_${seq_len}_${pred_len}"
    log_file="logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"

    python -u run_exp.py \
      --random_seed "${random_seed}" \
      --is_training 1 \
      --model_id "${model_id}" \
      --model "${model_name}" \
      --data "${data_name}" \
      --root_path "${root_path}" \
      --features M \
      --seq_len "${seq_len}" \
      --pred_len "${pred_len}" \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 128\
      --d_ff 512 \
      --d_hub 1 \
      --dropout 0.2\
      --head_dropout 0 \
      --train_epochs 20 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0003\
      > "${log_file}" 2>&1

    echo "Finished run: ${model_id}, log -> ${log_file}"
done
for pred_len in 336 720 
do
    model_id="${model_id_name}_${seq_len}_${pred_len}"
    log_file="logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"

    python -u run_exp.py \
      --random_seed "${random_seed}" \
      --is_training 1 \
      --model_id "${model_id}" \
      --model "${model_name}" \
      --data "${data_name}" \
      --root_path "${root_path}" \
      --features M \
      --seq_len "${seq_len}" \
      --pred_len "${pred_len}" \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 64 \
      --d_ff 256 \
      --d_hub 1 \
      --dropout 0.2 \
      --head_dropout 0 \
      --train_epochs 20 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001\
      > "${log_file}" 2>&1

    echo "Finished run: ${model_id}, log -> ${log_file}"
done
