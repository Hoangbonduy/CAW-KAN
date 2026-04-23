# Lấy đường dẫn gốc
model_name=CAW_KAN

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Chạy thử nghiệm CAW_KAN - Cấu hình chống Overfit cực đoan cho Illness
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id illness_36_24 \
  --model $model_name \
  --data illness \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --freq w \
  --features M \
  --target OT \
  --seq_len 36 \
  --label_len 0 \
  --pred_len 24 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --n_heads 4 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 64 \
  --factor 1 \
  --embed timeF \
  --dropout 0.2 \
  --use_amp \
  --channel_independence 1 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --patience 10 \
  --weight_decay 1e-3 \
  --lradj 'cosine' \
  --pct_start 0.2 \
  --des 'Exp_CAW_KAN_researching_Tuned'