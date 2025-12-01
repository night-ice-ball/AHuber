#!/usr/bin/env bash

mkdir -p ./logs/LongForecasting

seq_len=96
model_name=AHuber

root_path=./data/electricity
data_path_name=electricity.csv
model_id_name=electricity
data_name=electricity

random_seed=2021

for pred_len in 96 192 336 720
do
    model_id="${model_id_name}_${seq_len}_${pred_len}"
    log_file="logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"

    python -u gogogo.py \
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
      --e_layers 3 \
      --n_heads 8 \
      --d_model 128 \
      --d_ff 512 \
      --dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 16 \
      --train_epochs 25 \
      --itr 1 \
      --batch_size 32 \
      --learning_rate 0.001 \
      > "${log_file}" 2>&1

    echo "Finished run: ${model_id}, log -> ${log_file}"
done
