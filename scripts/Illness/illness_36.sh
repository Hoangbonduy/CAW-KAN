#!/bin/bash

# Lấy đường dẫn gốc
model_name=CAW_KAN
wavelet_type=mexican_hat
num_wavelets=8
kernel_size=7

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

echo "====================================================================="
echo "🚀 Bắt đầu chạy CAW_KAN - Dataset: Illness - Pred Len: 24"
echo "Cấu hình tối ưu: d_model=32, epochs=50, patience=15, batch_size=16"
echo "====================================================================="

# Chạy thử nghiệm với cấu hình đã tối ưu để trị Overfitting & Valid Loss stuck
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id illness_104_24_optimized \
  --model $model_name \
  --data custom \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --features M \
  --seq_len 104 \
  --label_len 0 \
  --pred_len 36 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --n_heads 4 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 32 \
  --factor 1 \
  --embed timeF \
  --dropout 0.1 \
  --use_amp \
  --channel_independence 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 15 \
  --lradj 'cosine' \
  --des 'Fix_ValidLoss_Stuck' \
  --itr 1