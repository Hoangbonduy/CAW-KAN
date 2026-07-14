import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from thop import profile, clever_format

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from layers.AdaptiveWaveletKAN import AdaptiveWaveletKANLayer
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from models.CAW_KAN import ContextAwareWavKANBlock, Model

BATCH_SIZE = 1

FREQ_TIME_DIM = {
    "h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3,
}

COMMON_CONFIG = {
    "model": "CAW_KAN",
    "task_name": "long_term_forecast",
    "features": "M",
    "embed": "timeF",
    "freq": "h",
    "seq_len": 512,
    "label_len": 0,
    "pred_len": 96,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 16,
    "d_ff": 32,
    "factor": 1,
    "dropout": 0.1,
    "wavelet_type": "mexican_hat",
    "grid_size": 3.0,
    "channel_independence": 1,
    "batch_size": BATCH_SIZE,
}

# Đã bổ sung "d_model": 32 cho ETTm2 để đồng bộ tính toán với TimeKAN
DATASET_CONFIGS = [
    {"data": "ETTh1", "model_id": "ETTh1", "e_layers": 2, "num_wavelets": 4, "kernel_size": 3},
    {"data": "ETTh2", "model_id": "ETTh2", "e_layers": 3, "num_wavelets": 4, "kernel_size": 3},
    {"data": "ETTm1", "model_id": "ETTm1", "e_layers": 2, "freq": "t", "num_wavelets": 4, "kernel_size": 7},
    {"data": "ETTm2", "model_id": "ETTm2", "e_layers": 3, "freq": "t", "num_wavelets": 4, "grid_size": 4.0, "kernel_size": 7, "d_model": 32},
    {"data": "weather", "model_id": "weather", "e_layers": 3, "freq": "t", "num_wavelets": 4, "kernel_size": 3, "enc_in": 21, "dec_in": 21, "c_out": 21}
]

class ProfileWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_enc, x_mark_enc):
        return self.model(x_enc, x_mark_enc, None, None)

def _to_namespace(config_dict):
    merged = dict(COMMON_CONFIG)
    merged.update(config_dict)
    return SimpleNamespace(**merged)

def _time_feature_dim(freq: str):
    freq_key = str(freq).lower()
    if freq_key in FREQ_TIME_DIM: return FREQ_TIME_DIM[freq_key]
    if freq_key.endswith("min"): return FREQ_TIME_DIM["t"]
    if freq_key and freq_key[-1] in FREQ_TIME_DIM: return FREQ_TIME_DIM[freq_key[-1]]
    return FREQ_TIME_DIM["h"]

def _build_dummy_x_mark(config):
    if getattr(config, "embed", "timeF") == "timeF":
        time_dim = _time_feature_dim(config.freq)
        return torch.randn(config.batch_size, config.seq_len, time_dim)
    return torch.zeros(config.batch_size, config.seq_len, 5, dtype=torch.long)

def _profile_input_seq_len(config):
    block_count = int(getattr(config, "e_layers", 1))
    kernel_size = int(getattr(config, "kernel_size", 1))
    shrink_per_block = max(kernel_size - 1, 0)
    return int(config.seq_len + block_count * shrink_per_block)

# =====================================================================
# HÀM HOOK CUSTOM: LƯU TRỮ MACs TRÁNH BỊ THOP XÓA
# =====================================================================
def _store_macs(module, macs):
    macs = int(macs)
    module.total_ops += torch.DoubleTensor([macs])
    module.stored_macs = getattr(module, "stored_macs", 0) + macs

def count_adaptive_wavelet_kan(m, x, y):
    x_in = x[0]
    num_elements = x_in.numel() 
    macs_per_element_per_wavelet = 4.5
    total_macs = num_elements * m.num_wavelets * macs_per_element_per_wavelet
    _store_macs(m, total_macs)

def count_conv1d_custom(m, x, y):
    out_elements = y.numel()
    kernel_size = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
    groups = max(int(m.groups), 1)
    total_macs = out_elements * kernel_size * (m.in_channels // groups)
    _store_macs(m, total_macs)

def count_linear_custom(m, x, y):
    total_macs = y.numel() * m.in_features
    _store_macs(m, total_macs)

def count_embedding_add_custom(m, x, y):
    total_macs = y.numel() if len(x) > 1 and x[1] is not None else 0
    _store_macs(m, total_macs)

def count_block_aux_custom(m, x, y):
    total_macs = 2 * y.numel()
    _store_macs(m, total_macs)

def count_normalize_custom(m, x, y):
    ops_per_element = 8 if m.affine else 6
    total_macs = y.numel() * ops_per_element
    _store_macs(m, total_macs)
# =====================================================================

def profile_one(config):
    model = Model(config)
    model.eval()

    wrapped_model = ProfileWrapper(model)
    profile_seq_len = _profile_input_seq_len(config)
    dummy_x = torch.randn(config.batch_size, profile_seq_len, config.enc_in)
    dummy_x_mark = _build_dummy_x_mark(config)
    if dummy_x_mark.shape[1] != profile_seq_len:
        dummy_x_mark = torch.randn(dummy_x_mark.shape[0], profile_seq_len, dummy_x_mark.shape[-1])
        
    total_params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        profile(
            wrapped_model, 
            inputs=(dummy_x, dummy_x_mark), 
            custom_ops={
                AdaptiveWaveletKANLayer: count_adaptive_wavelet_kan,
                nn.Conv1d: count_conv1d_custom,
                nn.Linear: count_linear_custom,
                DataEmbedding_wo_pos: count_embedding_add_custom,
                ContextAwareWavKANBlock: count_block_aux_custom,
                Normalize: count_normalize_custom,
            },
            verbose=False
        )

    embedding_macs = 0
    predictor_macs = 0
    projector_macs = 0
    cawkan_macs = 0
    remaining_macs = 0

    for name, module in wrapped_model.named_modules():
        stored_macs = getattr(module, "stored_macs", 0)
        if not stored_macs:
            continue

        if isinstance(module, DataEmbedding_wo_pos) or ("enc_embedding" in name):
            embedding_macs += stored_macs
        elif "predictor" in name:
            predictor_macs += stored_macs
        elif "projector" in name:
            projector_macs += stored_macs
        elif isinstance(module, AdaptiveWaveletKANLayer) or isinstance(module, nn.Conv1d):
            cawkan_macs += stored_macs
        elif isinstance(module, ContextAwareWavKANBlock) or isinstance(module, Normalize) or isinstance(module, nn.LayerNorm):
            remaining_macs += stored_macs

    macs_total = embedding_macs + predictor_macs + projector_macs + cawkan_macs + remaining_macs
    macs_str = clever_format([macs_total], "%.3f")

    return {
        "dataset": config.data,
        "batch_size": config.batch_size,
        "params_total": total_params,
        "macs_embedding": embedding_macs,
        "macs_predictor": predictor_macs,
        "macs_projector": projector_macs,
        "macs_cawkan": cawkan_macs,
        "macs_remaining": remaining_macs,
        "macs_total": macs_total,
        "macs_str": macs_str,
    }

def print_results(rows):
    header = (
        f"{'Dataset':<8} {'Batch':<6} {'Params(total)':>14} "
        f"{'Embed MAC':>12} {'Predictor':>12} {'Projector':>12} {'CAW-KAN':>12} {'Còn lại':>12} {'Tổng':>12} {'Format':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['dataset']:<8} "
            f"{row['batch_size']:<6d} "
            f"{int(row['params_total']):>14,} "
            f"{int(row['macs_embedding']):>12,} "
            f"{int(row['macs_predictor']):>12,} "
            f"{int(row['macs_projector']):>12,} "
            f"{int(row['macs_cawkan']):>12,} "
            f"{int(row['macs_remaining']):>12,} "
            f"{int(row['macs_total']):>12,} "
            f"{row['macs_str']:>9}"
        )

def main():
    rows = []
    for cfg in DATASET_CONFIGS:
        config = _to_namespace(cfg)
        rows.append(profile_one(config))
    print_results(rows)

if __name__ == "__main__":
    main()