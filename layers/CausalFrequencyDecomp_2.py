import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class DWTFrequencyDecomp(nn.Module):
    def __init__(self, wave='haar', J=1):
        """
        wave: Loại Wavelet sử dụng (ví dụ: 'haar', 'db2', 'db4', ...). 
              'haar' tương đương với cửa sổ trượt sắc nét, 'db' mượt mà hơn.
        J: Số mức phân rã (Decomposition Level). J=1 là chia 1 lần (Trend/Spike cơ bản).
        """
        super().__init__()
        self.wave = wave
        self.J = J
        
        # Khởi tạo DWT (Forward) và Inverse DWT (Backward)
        # mode='zero' kết hợp với causal padding bên dưới để tránh Data Leakage
        self.dwt = DWT1DForward(wave=wave, J=J, mode='zero')
        self.idwt = DWT1DInverse(wave=wave, mode='zero')

    def forward(self, x):
        # x: [Batch, Seq, Channel]
        x_in = x.permute(0, 2, 1) # -> [Batch, Channel, Seq]
        seq_len = x_in.shape[-1]
        
        # BƯỚC 1: Padding nhân quả (Causal Padding)
        # DWT thường làm ngắn chuỗi. Ta pad số 0 bên trái để đảm bảo:
        # 1. Tính nhân quả (không nhìn vào tương lai)
        # 2. Đủ độ dài để DWT không bị lỗi biên (đặc biệt khi seq_len lẻ)
        pad_len = (2 ** self.J) - (seq_len % (2 ** self.J)) if seq_len % (2 ** self.J) != 0 else 0
        if pad_len > 0:
            x_pad = F.pad(x_in, (pad_len, 0), mode='constant', value=0.0)
        else:
            x_pad = x_in
            
        # BƯỚC 2: Phân rã DWT
        # yl: Low-frequency (Hệ số xấp xỉ - Approximation / Trend)
        # yh: High-frequency (Hệ số chi tiết - Detail / Spike)
        yl, yh = self.dwt(x_pad)
        
        # BƯỚC 3: Trích xuất Trend (Khôi phục dải tần thấp về miền thời gian)
        # Bằng cách ép toàn bộ hệ số tần số cao (yh) về 0, IDWT sẽ chỉ khôi phục lại Trend.
        yh_zeros = [torch.zeros_like(h) for h in yh]
        x_trend_pad = self.idwt((yl, yh_zeros))
        
        # BƯỚC 4: Loại bỏ phần Padding để trả về kích thước gốc
        # Do pad bên trái, ta lấy dữ liệu từ phần tử thứ pad_len trở đi
        x_trend = x_trend_pad[..., pad_len:pad_len + seq_len]
        
        # Đảm bảo shape khớp tuyệt đối (phòng sai số làm tròn của pytorch_wavelets)
        if x_trend.shape[-1] > seq_len:
            x_trend = x_trend[..., :seq_len]
        elif x_trend.shape[-1] < seq_len:
            x_trend = F.pad(x_trend, (0, seq_len - x_trend.shape[-1]))
            
        # BƯỚC 5: Tín hiệu Spike
        # Về mặt toán học: X_gốc = X_trend + X_spike. 
        # Cách tách sạch nhất và đảm bảo bảo toàn thông tin 100% là dùng phép trừ trực tiếp.
        x_res = x_in - x_trend
        
        # Output trả về đúng định dạng [Batch, Seq, Channel] cho PhaseAwareJDKANBlock
        return x_trend.permute(0, 2, 1), x_res.permute(0, 2, 1)