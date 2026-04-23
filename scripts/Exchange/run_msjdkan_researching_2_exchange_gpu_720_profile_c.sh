# Profile C: near-baseline stabilization for pred_len=720
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
  --model_id exchange_96_720_profile_c \
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
  --d_model 32 \
  --n_heads 4 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 32 \
  --factor 1 \
  --embed timeF \
  --dropout 0.12 \
  --use_amp \
  --channel_independence 1 \
  --batch_size 32 \
  --learning_rate 0.0007 \
  --train_epochs 120 \
  --patience 15 \
  --weight_decay 1e-4 \
  --grad_clip 0.7 \
  --lradj 'cosine' \
  --pct_start 0.2 \
  --des Exp_CAW_KAN_researching_profile_c
