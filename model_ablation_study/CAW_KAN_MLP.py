import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.StandardNorm import Normalize
# from layers.AdaptiveWaveletKAN import AdaptiveWaveletKANLayer
from layers.StandardMLPLayer import StandardMLPLayer
from layers.Embed import DataEmbedding_wo_pos

class ContextAwareMLPBlock(nn.Module): # Đổi tên Block cho phù hợp với Ablation
    def __init__(self, d_model, seq_len, dropout=0.1, num_wavelets=8, wavelet_type='mexican_hat', grid_size=3.0,
                 kernel_size=7):
        super().__init__()
        self.d_model = d_model
        
        # 1. Khảm Ngữ cảnh thông qua Tích chập 1D
        self.context_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2)

        # --- 2. Lõi Standard MLP (THAY THẾ KAN) ---
        self.core_mlp = StandardMLPLayer(d_model, d_model, seq_len, num_wavelets=num_wavelets)
        
        # 3. Residual & Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, debug_store=None, block_idx=None):
        # --- BƯỚC 1: Nhúng ngữ cảnh cục bộ ---
        x_context = x.transpose(1, 2)
        x_context = self.context_conv(x_context)
        x_context = x_context.transpose(1, 2)

        if debug_store is not None and block_idx is not None:
            debug_store[f"Block_{block_idx:02d}_After_ContextConv"] = x_context.clone()

        x_context = self.norm1(x + x_context) 

        if debug_store is not None and block_idx is not None:
            debug_store[f"Block_{block_idx:02d}_After_Norm1"] = x_context.clone()
        
        # --- BƯỚC 2: Truyền qua Standard MLP thay vì KAN ---
        mlp_out = self.core_mlp(x_context)
        
        # --- BƯỚC 3: Tính Residual cho Block sau ---
        next_x = self.norm2(x_context + mlp_out)
        next_x = self.dropout(next_x)

        if debug_store is not None and block_idx is not None:
            debug_store[f"Block_{block_idx:02d}_MLP_Out"] = mlp_out.clone()
            debug_store[f"Block_{block_idx:02d}_Next_X"] = next_x.clone()

        return mlp_out, next_x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.kernel_size = configs.kernel_size
        
        self.enc_embedding = DataEmbedding_wo_pos(c_in=1, d_model=configs.d_model, freq = getattr(configs, 'freq', 'h'), dropout=configs.dropout)
        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False, subtract_last=False)
        
        # --- Cập nhật danh sách Blocks sử dụng MLP ---
        self.blocks = nn.ModuleList([
            ContextAwareMLPBlock(
                d_model=configs.d_model,
                seq_len=self.seq_len,
                dropout=configs.dropout,
                num_wavelets=configs.num_wavelets,
                kernel_size=self.kernel_size
            ) for _ in range(configs.e_layers)
        ])
        
        self.projector = nn.Linear(configs.d_model, 1)
        self.predictor = nn.Linear(self.seq_len, self.pred_len)
        
    def forecast(self, x_enc, x_mark_enc=None, debug_store=None):
        # [Giữ nguyên code phần forecast giống như cũ, chỉ đổi tên biến nếu cần cho dễ đọc]
        x_norm = self.normalize_layer(x_enc, 'norm')
        B, T, C = x_norm.shape
        x_reshaped = x_norm.permute(0, 2, 1).contiguous().reshape(B * C, T, 1)
        
        if x_mark_enc is not None:
            x_mark_reshaped = x_mark_enc.unsqueeze(1).repeat(1, C, 1, 1).reshape(B * C, T, -1)
        else:
            x_mark_reshaped = None
            
        enc_out = self.enc_embedding(x_reshaped, x_mark_reshaped)
        curr_input = enc_out

        for i, block in enumerate(self.blocks):
            core_out, next_input = block(curr_input, debug_store=debug_store, block_idx=i)
            curr_input = next_input
            
        dec_out = self.predictor(curr_input.transpose(1, 2)).transpose(1, 2)
        dec_out = self.projector(dec_out) 

        dec_out = dec_out.reshape(B, C, self.pred_len).permute(0, 2, 1)
        dec_out = self.normalize_layer(dec_out, 'denorm')
        
        if debug_store is not None:
            debug_store["Final_Output_Projection"] = dec_out.clone()
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, debug_store=None):
        return self.forecast(x_enc, x_mark_enc=x_mark_enc, debug_store=debug_store)