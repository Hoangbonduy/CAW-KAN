## Cài đặt môi trường

```bash
cd CAW-KAN
conda create -n cawkan python=3.12
conda activate cawkan
pip install -r requirements.txt
```

## Dataset

Dataset lấy từ [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

Sau khi tải xong, đặt dữ liệu vào thư mục `dataset` của project.

## Chạy các dữ liệu

### Etth1 với độ dài dự báo 96

```bash
bash scripts/Etth1/etth1_96.sh
```

### Etth1 với độ dài dự báo 96 không dùng gpu
```bash
bash scripts/Etth1/etth1_96.sh --no_use_gpu
```