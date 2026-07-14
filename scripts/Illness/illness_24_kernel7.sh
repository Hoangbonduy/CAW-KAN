# Lấy đường dẫn gốc
model_name=CAW_KAN
wavelet_type=mexican_hat
num_wavelets=8
kernel_size=7

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
    --model_id illness_96_24 \
  --model $model_name \
    --data custom \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
  --features M \
  --seq_len 104 \
  --label_len 0 \
  --pred_len 24 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 256 \
  --factor 1 \
  --embed timeF \
  --dropout 0.1 \
  --use_amp \
  --channel_independence 1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --weight_decay 1e-4 \
  --lradj 'cosine' \
  --pct_start 0.2 \
  --wavelet_type $wavelet_type \
  --num_wavelets $num_wavelets \
  --kernel_size $kernel_size \
  --des Exp_CAW_KAN_researching