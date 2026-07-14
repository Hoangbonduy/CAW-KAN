import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        Nâng cấp: RevIN đầy đủ (Normalize cả Mean và Variance)
        Theo khuyến nghị từ báo cáo phân tích lỗi MS_JDKAN.
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        
        if self.affine:
            self._init_params()

    def _init_params(self):
        # Learnable parameters: gamma (scale) và beta (shift)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        # 1. Tính Mean (Level)
        if self.subtract_last:
            # Nếu dùng subtract_last, Mean chính là giá trị cuối cùng
            self.mean = x[:, -1, :].unsqueeze(1)
        else:
            # Nếu không, dùng trung bình cộng
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            
        # 2. Tính Stdev (Scale) - ĐÂY LÀ BƯỚC QUAN TRỌNG MỚI THÊM VÀO
        # Báo cáo[cite: 179]: Phải tính variance để chuẩn hóa biên độ
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
            
        # Bước 1: Trừ Mean (Level Normalization)
        x = x - self.mean
        
        # Bước 2: Chia Stdev (Variance Normalization) - FIX LỖI OVERFIT VALIDATION
        # Báo cáo[cite: 182]: Đưa dữ liệu về phạm vi hoạt động của Wavelet
        x = x / self.stdev
        
        # Bước 3: Affine (nếu có)
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
            
        # Bước 1: Đảo ngược Affine
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            
        # Bước 2: Nhân lại Stdev (Khôi phục biên độ) - QUAN TRỌNG
        x = x * self.stdev
        
        # Bước 3: Cộng lại Mean (Khôi phục xu hướng)
        x = x + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x