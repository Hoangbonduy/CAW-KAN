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
        
        # --- Nhánh Wavelet ---
        self.w = nn.Parameter(torch.empty(in_features, num_wavelets))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        
        if self.wavelet_type == 'morlet':
            self.register_buffer('omega0', torch.tensor(5.0))
        else:
            self.register_buffer('omega0', None)
        
        grid_min, grid_max = -grid_size, grid_size

        # --- KHỞI TẠO LƯỚI ĐƠN (SINGLE-GRID ABLATION) ---
        
        # 1. Rải đều toàn bộ tâm sóng (b) trên toàn miền giới hạn [cite: 51, 60]
        base_b = torch.linspace(grid_min, grid_max, num_wavelets)
        
        # 2. Tính khoảng cách step giữa các tâm sóng lân cận
        # (Dùng tổng số sóng thay vì chia nhóm như trước) [cite: 57]
        step = (grid_max - grid_min) / (num_wavelets - 1)
        
        # 3. Áp dụng hệ số giãn nở (a) cho toàn bộ sóng
        # Giữ nguyên tỷ lệ 0.8 * step để đảm bảo tính liên tục của hàm sóng (đặc biệt là Mexican Hat) 
        base_a = torch.ones(num_wavelets) * step * 0.8
        
        # --- Tổng hợp Grid ---
        grid_b = base_b.unsqueeze(0).repeat(in_features, 1) 
        grid_a = base_a.unsqueeze(0).repeat(in_features, 1)
        
        self.register_buffer('b', grid_b.view(1, 1, in_features, num_wavelets))
        self.register_buffer('a', grid_a.view(1, 1, in_features, num_wavelets))
    
    def _compute_wavelet_response(self, z):
        if self.wavelet_type == 'mexican_hat':
            # coeff = 2.0 / (math.sqrt(3.0) * (math.pi ** 0.25))
            # return coeff * (z**2 - 1.0) * torch.exp(-0.5 * z**2)
            # return coeff * (z**2 - 1.0) * torch.exp(-0.5 * z**2)
            return (1.0 - z**2) * torch.exp(-0.5 * z**2)  # Bỏ hệ số chuẩn hóa để tăng biên độ, giúp mạng dễ học hơn

        if self.wavelet_type == 'morlet':
            return torch.cos(self.omega0 * z) * torch.exp(-0.5 * z**2)

        if self.wavelet_type == 'dog':
            return z * torch.exp(-0.5 * z**2)

        if self.wavelet_type == 'shannon':
            window = (z.abs() <= math.pi).to(z.dtype)
            return torch.sinc(z / math.pi) * window

    def forward(self, x):
        # x input: [Batch, Seq, Channel]
        
        # --- Chỉ tính toán một nhánh Wavelet duy nhất ---
        x_expanded = x.unsqueeze(-1) # -> [Batch, Seq, Channel, 1]
        
        # Tránh chia cho 0
        z = (x_expanded - self.b) / (torch.abs(self.a) + 1e-6)
        psi = self._compute_wavelet_response(z)
        
        w = self.w.view(1, 1, self.in_features, self.num_wavelets)
        
        # Output trực tiếp từ tổ hợp tuyến tính của Wavelet
        out = (w * psi).sum(dim=-1) # Output: [Batch, Seq, Channel]
        
        return out