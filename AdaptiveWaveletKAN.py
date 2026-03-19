import torch
import torch.nn as nn
import math

class AdaptiveWaveletKANLayer(nn.Module):
    def __init__(self, in_features, out_features, seq_len, num_wavelets=7):
        super(AdaptiveWaveletKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.num_wavelets = num_wavelets
        
        # --- Nhánh Wavelet ---
        self.w = nn.Parameter(torch.empty(in_features, num_wavelets))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        
        grid_min, grid_max = -3.0, 3.0

        # Làm tròn lên cho Trend, phần còn lại cho Detail
        num_trend = (num_wavelets + 1) // 2   # 7 -> 4
        num_detail = num_wavelets - num_trend  # 7 -> 3

        # --- Nhánh Trend: trải đều trên toàn miền ---
        b_trend = torch.linspace(grid_min, grid_max, num_trend)  # [-3, -1, 1, 3]
        step = (grid_max - grid_min) / (num_trend - 1)           # step = 2.0
        a_trend = torch.ones(num_trend) * step * 0.8             # a = 2.0

        # --- Nhánh Detail: so le (lấp đúng khe giữa các wavelet trend) ---
        detail_min = grid_min + step / 2  # -3.0 + 1.0 = -2.0
        detail_max = grid_max - step / 2  #  3.0 - 1.0 =  2.0
        b_detail = torch.linspace(detail_min, detail_max, num_detail)  # [-2, 0, 2]
        a_detail = torch.ones(num_detail) * step * 0.4                 # a = 1.0

        # --- Tổng hợp Grid ---
        base_b = torch.cat([b_trend, b_detail], dim=0)
        grid_b = base_b.unsqueeze(0).repeat(in_features, 1) 
        
        base_a = torch.cat([a_trend, a_detail], dim=0)
        grid_a = base_a.unsqueeze(0).repeat(in_features, 1)
        
        # Thử để nn.Parameter thay vì register_buffer để mạng tự fine-tune nhẹ lưới
        # self.b = nn.Parameter(grid_b.view(1, 1, in_features, num_wavelets))
        # self.a = nn.Parameter(grid_a.view(1, 1, in_features, num_wavelets))
        self.register_buffer('b', grid_b.view(1, 1, in_features, num_wavelets))
        self.register_buffer('a', grid_a.view(1, 1, in_features, num_wavelets))

    def forward(self, x):
        # x input: [Batch, Seq, Channel]
        
        # --- Chỉ tính toán một nhánh Wavelet duy nhất ---
        x_expanded = x.unsqueeze(-1) # -> [Batch, Seq, Channel, 1]
        
        # Tránh chia cho 0
        z = (x_expanded - self.b) / (torch.abs(self.a) + 1e-6)
        
        # MexicanHat Wavelet
        psi = (1 - z**2) * torch.exp(-0.5 * z**2)
        
        w = self.w.view(1, 1, self.in_features, self.num_wavelets)
        
        # Output trực tiếp từ tổ hợp tuyến tính của Wavelet
        out = (w * psi).sum(dim=-1) # Output: [Batch, Seq, Channel]
        
        return out