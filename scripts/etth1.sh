#!/usr/bin/env bash

mkdir -p ./logs

seq_len=96
model_name=AHuber
root_path=./data/ETT
model_id_name=ETTh1
data_name=ETTh1
random_seed=2021

for pred_len in 96 192 336 720
do
    model_id="${model_id_name}_${seq_len}_${pred_len}"
    log_file="logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"

    python -u gogogo.py \
      --random_seed "${random_seed}" \
      --is_training 1 \
      --model_id "${model_id}" \
      --model "${model_name}" \
      --data "${data_name}" \
      --root_path "${root_path}" \
      --features M \
      --seq_len "${seq_len}" \
      --pred_len "${pred_len}" \
      --e_layers 2 \
      --n_heads 4 \
      --d_model 64 \
      --d_ff 128 \
      --d_hub 1 \
      --dropout 0.3 \
      --head_dropout 0 \
      --train_epochs 15 \
      --itr 1 \
      --batch_size 64 \
      --learning_rate 0.001 \
      > "${log_file}" 2>&1

    echo "Finished run: ${model_id}, log -> ${log_file}"
done
