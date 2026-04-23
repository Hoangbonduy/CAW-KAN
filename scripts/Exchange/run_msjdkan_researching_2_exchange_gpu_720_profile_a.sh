# Profile A: strong stabilization for pred_len=720
model_name=CAW_KAN

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id exchange_96_720_profile_a \
  --model $model_name \
  --data exchange_rate\
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --freq d \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 16 \
  --n_heads 4 \
  --e_layers 4 \
  --d_layers 1 \
  --d_ff 32 \
  --factor 1 \
  --embed timeF \
  --dropout 0.2 \
  --use_amp \
  --channel_independence 1 \
  --batch_size 128 \
  --learning_rate 0.0003 \
  --train_epochs 150 \
  --patience 20 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lradj 'TST' \
  --pct_start 0.25 \
  --des Exp_CAW_KAN_researching_profile_a
