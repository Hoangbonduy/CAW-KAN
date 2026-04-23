# Lấy đường dẫn gốc
model_name=MS_JDKAN
wavelet_type=mexican_hat
look_back_window_list="48"
num_wavelets=8
grid_size=3.0
kernel_size=3

# Ban đầu d_model = 32, d_ff = 64

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Chạy thử nghiệm MS_JDKAN
for look_back_window in $look_back_window_list; do
  echo "===== Running look_back_window=$look_back_window ====="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_lb${look_back_window}_96 \
  --model $model_name \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --target OT \
  --seq_len $look_back_window \
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
  --grid_size $grid_size \
  --kernel_size $kernel_size \
  --des Exp_MS_JDKAN_lookback_lb${look_back_window}
done