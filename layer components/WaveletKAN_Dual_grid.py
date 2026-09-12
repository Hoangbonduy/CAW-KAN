import torch
import torch.nn as nn
import math

class AdaptiveWaveletKANLayer(nn.Module):
    def __init__(self, in_features, out_features, seq_len, num_wavelets=7, wavelet_type='mexican_hat', grid_size=3.0):
        super(AdaptiveWaveletKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type.lower()
        
        valid_wavelets = {'mexican_hat', 'morlet', 'dog', 'shannon'}
        if self.wavelet_type not in valid_wavelets:
            raise ValueError(f"Unsupported wavelet_type={wavelet_type}. Supported: {sorted(valid_wavelets)}")
        
        # --- 1. SỐ LƯỢNG WAVELET CHO DUAL-GRID ---
        # Phân bổ ngân sách cho Trend và Detail theo logic Dual-grid
        self.num_trend = (num_wavelets + 1) // 2   
        self.num_detail = num_wavelets - self.num_trend  

        # --- 2. KHỞI TẠO FULL TENSOR TRỌNG SỐ ---
        # Thay thế hoàn toàn CP-Factorization bằng 1 Full Tensor duy nhất
        self.weight = nn.Parameter(torch.empty(out_features, in_features, num_wavelets))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.wavelet_type == 'morlet':
            self.register_buffer('omega0', torch.tensor(5.0))
        else:
            self.register_buffer('omega0', None)
        
        # --- 3. KHỞI TẠO DUAL-GRID ---
        grid_min, grid_max = -grid_size, grid_size

        # Nhánh Trend: trải đều trên toàn miền
        b_trend = torch.linspace(grid_min, grid_max, self.num_trend)  
        step = (grid_max - grid_min) / max(self.num_trend - 1, 1)           
        a_trend = torch.ones(self.num_trend) * step * 0.8             

        # Nhánh Detail: offset nội bộ để lấp khe giữa các wavelet trend
        detail_min = grid_min + step / 2  
        detail_max = grid_max - step / 2  
        b_detail = torch.linspace(detail_min, detail_max, self.num_detail)  
        a_detail = torch.ones(self.num_detail) * step * 0.4                 

        # Tổng hợp Grid 
        base_b = torch.cat([b_trend, b_detail], dim=0)
        grid_b = base_b.unsqueeze(0).repeat(in_features, 1) 
        
        base_a = torch.cat([a_trend, a_detail], dim=0)
        grid_a = base_a.unsqueeze(0).repeat(in_features, 1)
        
        self.register_buffer('b', grid_b.view(1, 1, in_features, num_wavelets))
        self.register_buffer('a', grid_a.view(1, 1, in_features, num_wavelets))

    def _compute_wavelet_response(self, z):
        if self.wavelet_type == 'mexican_hat':
            return (1.0 - z**2) * torch.exp(-0.5 * z**2) 
        if self.wavelet_type == 'morlet':
            return torch.cos(self.omega0 * z) * torch.exp(-0.5 * z**2)
        if self.wavelet_type == 'dog':
            return z * torch.exp(-0.5 * z**2)
        if self.wavelet_type == 'shannon':
            window = (z.abs() <= math.pi).to(z.dtype)
            return torch.sinc(z / math.pi) * window

    def forward(self, x):
        # x: [B*C, Seq, in_features]
        x_expanded = x.unsqueeze(-1)
        z = (x_expanded - self.b) / (torch.abs(self.a) + 1e-6)
        
        # phi: [B*C, Seq, in_features, num_wavelets]
        phi = self._compute_wavelet_response(z)
        
        # --- FULL TENSOR FORWARD PASS ---
        # Nhân chập trực tiếp tín hiệu phi qua trọng số (không qua bước tách Block)
        # Sử dụng torch.einsum để xử lý gọn gàng các chiều batch và sequence
        out = torch.einsum('...iw, oiw -> ...o', phi, self.weight)
        
        return out