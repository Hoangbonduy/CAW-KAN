# Lấy đường dẫn gốc
model_name=CAW_KAN
degree=3
kernel_size=3

# Ban đầu d_model = 32, d_ff = 64

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Chạy thử nghiệm CAW_KAN
# Toán học đã tối ưu: Không cần khởi tạo grid_size, sử dụng Chebyshev đa tỷ lệ
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --target OT \
  --freq h \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --n_heads 4 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 32 \
  --factor 1 \
  --embed timeF \
  --dropout 0.1 \
  --channel_independence 1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --lradj 'cosine' \
  --pct_start 0.2 \
  --degree $degree \
  --kernel_size $kernel_size \
  --no_use_gpu \
  --des Exp_CAW_KAN_researching "$@"