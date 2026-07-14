# Lấy đường dẫn gốc
model_name=MS_JDKAN
wavelet_type=mexican_hat
num_wavelets=8
grid_size=3.0
kernel_sizes=(1 3 7 15)

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Chạy thử nghiệm MS_JDKAN
# Anti-overfit: giảm d_model, tăng dropout, thêm weight_decay, gradient clipping, cosine LR
for kernel_size in "${kernel_sizes[@]}"; do
  echo "Running with kernel_size=$kernel_size"
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
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
  --weight_decay 1e-4 \
  --lradj 'cosine' \
  --pct_start 0.2 \
  --wavelet_type $wavelet_type \
    --num_wavelets $num_wavelets \
  --grid_size $grid_size \
  --kernel_size $kernel_size \
  --des Exp_MS_JDKAN_researching

done
