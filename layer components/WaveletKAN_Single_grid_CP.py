import torch
import torch.nn as nn
import math

class AdaptiveWaveletKANLayer(nn.Module):
    def __init__(self, in_features, out_features, seq_len, num_wavelets=7, wavelet_type='mexican_hat', grid_size=3.0, rank=8):
        super(AdaptiveWaveletKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type.lower()
        self.rank = rank
        
        valid_wavelets = {'mexican_hat', 'morlet', 'dog', 'shannon'}
        if self.wavelet_type not in valid_wavelets:
            raise ValueError(f"Unsupported wavelet_type={wavelet_type}. Supported: {sorted(valid_wavelets)}")
        
        # --- 1. CP-FACTORIZATION BLOCK THỐNG NHẤT ---
        # Không chia rank_w hay rank_n, sử dụng trực tiếp rank và num_wavelets
        self.A = nn.Parameter(torch.empty(out_features, self.rank))
        self.B = nn.Parameter(torch.empty(in_features, self.rank))
        self.D = nn.Parameter(torch.empty(self.num_wavelets, self.rank))
        
        # Khởi tạo trọng số (Kaiming uniform)
        for param in [self.A, self.B, self.D]:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        
        if self.wavelet_type == 'morlet':
            self.register_buffer('omega0', torch.tensor(5.0))
        else:
            self.register_buffer('omega0', None)
        
        # --- 2. KHỞI TẠO SINGLE-GRID ---
        grid_min, grid_max = -grid_size, grid_size

        # Khởi tạo một grid duy nhất trải đều trên toàn miền cho tất cả wavelets
        base_b = torch.linspace(grid_min, grid_max, self.num_wavelets)  
        step = (grid_max - grid_min) / max(self.num_wavelets - 1, 1)           
        base_a = torch.ones(self.num_wavelets) * step * 0.8             

        # Tổng hợp Grid 
        grid_b = base_b.unsqueeze(0).repeat(in_features, 1) 
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
        
        # --- THU GỌN FORWARD PASS ---
        # Áp dụng CP-Factorization trên toàn bộ ma trận (thay vì tách 2 nhánh)
        U = torch.matmul(phi, self.D)           # Thu gọn mode-k -> [..., in_features, rank]
        v = torch.sum(U * self.B, dim=2)        # Thu gọn mode-i -> [..., rank]
        out = torch.matmul(v, self.A.t())       # Thu gọn mode-r -> [..., out_features]
        
        return out