import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.StandardNorm import Normalize
from layers.AdaptiveWaveletKAN import AdaptiveWaveletKANLayer
from layers.Embed import DataEmbedding_wo_pos

class ContextAwareWavKANBlock(nn.Module):
    """
    Block mới thay thế PhaseAwareJDKANBlock.
    Loại bỏ hoàn toàn J_list và tách trend thủ công.
    Tích hợp Conv1D để tạo ngữ cảnh (Context-Aware) trước khi đưa vào Wav-KAN.
    """
    def __init__(self, d_model, seq_len, dropout=0.1, num_wavelets=8):
        super().__init__()
        self.d_model = d_model
        
        # 1. Khảm Ngữ cảnh thông qua Tích chập 1D (Thay thế J_list)
        # Giúp hòa trộn thông tin chéo, cung cấp trường nhìn rộng cho Wavelet
        self.context_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        # 2. Lõi Adaptive Wavelet KAN (Xử lý trực tiếp tín hiệu nguyên bản)
        self.adaptive_kan = AdaptiveWaveletKANLayer(d_model, d_model, seq_len, num_wavelets=num_wavelets)
        
        # 3. Residual & Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) # Bắt buộc có chuẩn hóa sau KAN để kiểm soát biên độ
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, debug_store=None, block_idx=None):
        # x: Input gốc [Batch, Seq, D_model]
        
        # --- BƯỚC 1: Nhúng ngữ cảnh cục bộ (Contextualization) ---
        x_context = x.transpose(1, 2)
        x_context = self.context_conv(x_context)
        x_context = x_context.transpose(1, 2)

        if debug_store is not None and block_idx is not None:
            debug_store[f"Block_{block_idx:02d}_After_ContextConv"] = x_context.clone()

        x_context = self.norm1(x + x_context) # Residual connection 1

        if debug_store is not None and block_idx is not None:
            debug_store[f"Block_{block_idx:02d}_After_Norm1"] = x_context.clone()
        
        # --- BƯỚC 2: Truyền trực tiếp vào Wav-KAN ---
        # Mạng sẽ tự động điều chỉnh scale/translation để bắt cả Trend và Spike
        kan_out = self.adaptive_kan(x_context)
        
        # --- BƯỚC 3: Tính Residual cho Block sau ---
        next_x = self.norm2(x_context + kan_out) # Add & Norm
        next_x = self.dropout(next_x)

        if debug_store is not None and block_idx is not None:
            debug_store[f"Block_{block_idx:02d}_KAN_Out"] = kan_out.clone()
            debug_store[f"Block_{block_idx:02d}_Next_X"] = next_x.clone()

        return kan_out, next_x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # --- 1. Embedding ---
        self.enc_embedding = DataEmbedding_wo_pos(c_in=1, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout)
        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False, subtract_last=False)
        
        # --- 2. Encoder (Stacked Direct Wav-KAN Blocks) ---
        self.blocks = nn.ModuleList([
            ContextAwareWavKANBlock(
                d_model=configs.d_model,
                seq_len=self.seq_len,
                dropout=configs.dropout,
                num_wavelets=8 # Tăng số lượng wavelet để bù đắp việc bỏ J_list
            ) for _ in range(configs.e_layers)
        ])
        
        # --- 3. Output Projection ---
        self.projector = nn.Linear(configs.d_model, 1)
        self.predictor = nn.Linear(self.seq_len, self.pred_len)
        
    def forecast(self, x_enc, x_mark_enc=None, debug_store=None):
        # 1. Normalize
        x_norm = self.normalize_layer(x_enc, 'norm')
        
        # 2. Embedding
        B, T, C = x_norm.shape
        x_reshaped = x_norm.permute(0, 2, 1).contiguous().reshape(B * C, T, 1)
        if x_mark_enc is not None:
            x_mark_reshaped = x_mark_enc.unsqueeze(1).repeat(1, C, 1, 1).reshape(B * C, T, -1)
        else:
            x_mark_reshaped = None
        enc_out = self.enc_embedding(x_reshaped, x_mark_reshaped)
        
        curr_input = enc_out

        # 3. Encoding Loop
        for i, block in enumerate(self.blocks):
            kan_out, next_input = block(curr_input, debug_store=debug_store, block_idx=i)
            # Cập nhật đầu vào cho block tiếp theo (không cộng dồn kan_out nữa)
            curr_input = next_input
            
        # 4. Decoding / Projection
        # BƯỚC 1: Predict (Kéo dãn độ dài từ Seq_len sang Pred_len nhưng vẫn giữ không gian D_model)
        dec_out = self.predictor(curr_input.transpose(1, 2)).transpose(1, 2) # -> [Batch*Channel, Pred_Len, D_model]
        
        # BƯỚC 2: Project (Ép không gian D_model về 1 đặc trưng duy nhất để ra kết quả)
        dec_out = self.projector(dec_out) # -> [Batch*Channel, Pred_Len, 1]

        # 5. Reshape & Denormalize
        dec_out = dec_out.reshape(B, C, self.pred_len).permute(0, 2, 1)
        dec_out = self.normalize_layer(dec_out, 'denorm')
        
        if debug_store is not None:
            debug_store["Final_Output_Projection"] = dec_out.clone()
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, debug_store=None):
        return self.forecast(x_enc, x_mark_enc=x_mark_enc, debug_store=debug_store)