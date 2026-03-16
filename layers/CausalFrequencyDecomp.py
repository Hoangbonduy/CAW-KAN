import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class DWTFrequencyDecomp(nn.Module):
    def __init__(self, wave='db2', J=1): # Đổi mặc định sang db2
        super().__init__()
        self.wave = wave
        self.J = J
        
        self.dwt = DWT1DForward(wave=wave, J=J, mode='symmetric')
        self.idwt = DWT1DInverse(wave=wave, mode='symmetric')

    def forward(self, x):
        x_in = x.permute(0, 2, 1) # -> [Batch, Channel, Seq]
        seq_len = x_in.shape[-1]
        
        # Bước 1: Tính padding nhân quả (chỉ đệm bên trái)
        pad_len = (2 ** self.J) - (seq_len % (2 ** self.J)) if seq_len % (2 ** self.J) != 0 else 0
        if pad_len > 0:
            x_pad = F.pad(x_in, (pad_len, 0), mode='replicate')
        else:
            x_pad = x_in
            
        # Bước 2: Phân rã
        yl, yh = self.dwt(x_pad)
        
        # Bước 3: Khôi phục Trend
        yh_zeros = [torch.zeros_like(h) for h in yh]
        x_trend_pad = self.idwt((yl, yh_zeros))
        
        # Bước 4: Cắt chuỗi thông minh (Bao tiêu mọi loại sóng)
        # Bỏ qua đoạn if-else lằng nhằng cũ.
        # Dữ liệu gốc nằm ở cuối mảng, ta chỉ lấy đúng seq_len phần tử cuối cùng.
        x_trend = x_trend_pad[..., -seq_len:]
        
        # Bước 5: Tính phần dư (Spike)
        x_res = x_in - x_trend
        
        return x_trend.permute(0, 2, 1), x_res.permute(0, 2, 1)