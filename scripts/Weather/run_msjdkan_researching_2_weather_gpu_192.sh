# Lấy đường dẫn gốc
model_name=MS_JDKAN

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
    --model_id weather_96_192 \
  --model $model_name \
    --data weather\
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --freq t \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
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
  --weight_decay 1e-4 \
  --lradj 'cosine' \
  --pct_start 0.2 \
  --des Exp_MS_JDKAN_researching