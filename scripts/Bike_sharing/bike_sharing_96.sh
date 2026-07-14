# Lấy đường dẫn gốc
model_name=CAW_KAN
wavelet_type=mexican_hat
num_wavelets=8
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
# Anti-overfit: giảm d_model, tăng dropout, thêm weight_decay, gradient clipping, cosine LR
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id bike_sharing_512_96 \
  --model $model_name \
  --data custom \
  --root_path ./dataset/bike_sharing/ \
  --data_path bike_sharing.csv \
  --features M \
  --target OT \
  --freq h \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 16 \
  --n_heads 4 \
  --e_layers 3 \
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
  --kernel_size $kernel_size \
  --des Exp_CAW_KAN_researching