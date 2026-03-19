# Lấy đường dẫn gốc
model_name=MS_JDKAN
wavelet_type=dog

# Ban đầu d_model = 32, d_ff = 64

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Chạy thử nghiệm MS_JDKAN
# Anti-overfit: giảm d_model, tăng dropout, thêm weight_decay, gradient clipping, cosine LR
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --n_heads 4 \
  --e_layers 3 \
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
    --wavelet_type $wavelet_type \
  --des Exp_MS_JDKAN_researching