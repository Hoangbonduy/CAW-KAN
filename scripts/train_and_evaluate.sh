#!/bin/bash

# Mảng chứa thông tin các dataset: Tên_Dataset | Tên_File | Freq | Số_đặc_trưng | e_layers | grid_size | kernel_size
declare -a datasets=(
    "ETTh1 ETTh1.csv h 7 2 3.0 3"
    "ETTh2 ETTh2.csv h 7 3 3.0 3"
    "ETTm1 ETTm1.csv t 7 2 3.0 7"
    "ETTm2 ETTm2.csv t 7 3 4.0 7"
    "custom weather.csv 10m 21 3 3.0 3" # weather dùng data="custom"
)

# Các tham số gốc
model_name=CAW_KAN
wavelet_type=mexican_hat
num_wavelets=8

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

for ds_info in "${datasets[@]}"; do
    read -r data_name data_path freq enc_in e_layers grid_size kernel_size <<< "$ds_info"
    
    # Thiết lập lại thư mục root cho dataset
    if [ "$data_name" == "custom" ]; then
        model_id_prefix="Weather"
        root_path="./dataset/weather/"
    else
        model_id_prefix=$data_name
        root_path="./dataset/ETT-small/"
    fi

    echo "================ TRAIN & EVALUATE TRÊN $model_id_prefix ================"
    echo "Config: model=$model_name data=$data_name root=$root_path data_path=$data_path freq=$freq"
    echo "Config: enc_in=$enc_in e_layers=$e_layers wavelet_type=$wavelet_type num_wavelets=$num_wavelets"
    echo "Config: grid_size=$grid_size kernel_size=$kernel_size"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id ${model_id_prefix}_96_96 \
      --model $model_name \
      --data $data_name \
      --root_path $root_path \
      --data_path $data_path \
      --features M \
      --target OT \
      --freq $freq \
      --seq_len 512 \
      --label_len 0 \
      --pred_len 96 \
      --enc_in $enc_in \
      --dec_in $enc_in \
      --c_out $enc_in \
      --d_model 16 \
            --n_heads 4 \
            --e_layers $e_layers \
      --d_layers 1 \
      --d_ff 32 \
      --factor 1 \
      --dropout 0.1 \
      --channel_independence 1 \
      --batch_size 32 \
      --learning_rate 0.001 \
      --train_epochs 100 \
      --patience 10 \
      --lradj 'cosine' \
      --pct_start 0.2 \
      --wavelet_type $wavelet_type \
      --num_wavelets $num_wavelets \
      --grid_size $grid_size \
      --kernel_size $kernel_size \
      --des Exp_CAW_KAN_researching
      
    echo "Hoàn tất Train & Evaluate cho $model_id_prefix!"
    echo "--------------------------------------------------------"
done

echo "TẤT CẢ DATASET ĐÃ ĐƯỢC TRAIN VÀ ĐO LƯỜNG HIỆU NĂNG XONG!"