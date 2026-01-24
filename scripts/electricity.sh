#!/usr/bin/env bash

mkdir -p ./logs

seq_len=96
model_name=AHuber
root_path=./data/electricity
model_id_name=electricity
data_name=electricity
random_seed=2021

for pred_len in 96 192 336 720
do
    model_id="${model_id_name}_${seq_len}_${pred_len}"
    log_file="logs/${model_name}_${data_name}_${seq_len}_${pred_len}.log"

    python -u run_exp.py \
      --random_seed "${random_seed}" \
      --is_training 1 \
      --model_id "${model_id}" \
      --model "${model_name}" \
      --data "${data_name}" \
      --root_path "${root_path}" \
      --data_path "${data_path_name}" \
      --features M \
      --seq_len "${seq_len}" \
      --pred_len "${pred_len}" \
      --e_layers 4 \
      --n_heads 12 \
      --d_model 192 \
      --d_ff 384 \
      --dropout 0.2 \
      --head_dropout 0 \
      --d_hub 1 \
      --train_epochs 30 \
      --itr 1 \
      --batch_size 64 \
      --learning_rate 0.0005 \
      > "${log_file}" 2>&1

    echo "Finished run: ${model_id}, log -> ${log_file}"
done
