## Requirements

```bash
cd CAW-KAN
conda create -n cawkan python=3.12
conda activate cawkan
pip install -r requirements.txt
```

## Data Preparation

We use the datasets provided by the [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) repository for our experiments. 

Once you have downloaded the datasets, please create a `dataset` folder in the root directory of the project and put all the data files inside it.

## Chạy các dữ liệu

### Etth1 với độ dài dự báo 96

```bash
bash scripts/Etth1/etth1_96.sh
```

### Etth1 với độ dài dự báo 96 không dùng gpu
```bash
bash scripts/Etth1/etth1_96.sh --no_use_gpu
```