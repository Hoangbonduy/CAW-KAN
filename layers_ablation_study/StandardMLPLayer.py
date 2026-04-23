import torch
import torch.nn as nn

class StandardMLPLayer(nn.Module):
    def __init__(self, in_features, out_features, seq_len, num_wavelets=7, **kwargs):
        super(StandardMLPLayer, self).__init__()
        
        # Để so sánh công bằng về sức mạnh biểu diễn và số lượng tham số với WaveletKAN,
        # mở rộng không gian ẩn (hidden dimension) dựa trên num_wavelets.
        hidden_dim = in_features * num_wavelets 
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(), # Dùng GELU phổ biến trong các mô hình time-series hiện đại
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        # x input: [Batch, Seq, Channel/d_model]
        # Đầu ra có cùng shape với AdaptiveWaveletKANLayer
        out = self.mlp(x)
        return out