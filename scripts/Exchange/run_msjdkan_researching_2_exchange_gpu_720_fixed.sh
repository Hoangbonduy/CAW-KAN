# Baseline parameters for pred_len=720 + training loop fixes enabled via args
model_name=MS_JDKAN

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id exchange_96_720_fixed \
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
  --dropout 0.1 \
  --use_amp \
  --channel_independence 1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lradj 'cosine' \
  --pct_start 0.2 \
  --des Exp_MS_JDKAN_researching_fixed
