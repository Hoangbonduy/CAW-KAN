import torch
import torch.nn as nn
import math

class ChebyshevKANLayer(nn.Module):
    def __init__(self, in_features, out_features, seq_len, degree=7):
        """
        Lớp KAN sử dụng Đa thức Chebyshev thay cho Wavelet.
        degree: Tương đương với num_wavelets cũ, xác định bậc tối đa của đa thức.
                Bậc càng cao, độ phi tuyến càng mạnh (nhưng cũng tăng rủi ro bắt nhiễu).
        """
        super(ChebyshevKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features # Giữ nguyên để tương thích API cũ
        self.degree = degree
        
        # Khởi tạo trọng số cho từng bậc của đa thức (từ bậc 0 đến degree)
        # Shape: [in_features, degree + 1]
        self.w = nn.Parameter(torch.empty(in_features, degree + 1))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):
        # x input: [Batch, Seq, Channel]
        
        # Chebyshev hoạt động chuẩn nhất khi input nằm trong khoảng [-1, 1].
        # Sử dụng tanh để chuẩn hóa và ép tín hiệu (bao gồm cả nhiễu) vào dải an toàn.
        x_norm = torch.tanh(x)
        
        # Khởi tạo danh sách chứa các đa thức T_0(x) = 1 và T_1(x) = x
        T = [torch.ones_like(x_norm), x_norm]
        
        # Tính các bậc tiếp theo bằng công thức truy hồi
        # T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
        for i in range(2, self.degree + 1):
            T_next = 2 * x_norm * T[i-1] - T[i-2]
            T.append(T_next)
            
        # Stack lại thành shape: [Batch, Seq, Channel, degree + 1]
        T_stacked = torch.stack(T, dim=-1)
        
        # Reshape trọng số w để nhân element-wise với T_stacked
        # w_view shape: [1, 1, Channel, degree + 1]
        w_view = self.w.view(1, 1, self.in_features, self.degree + 1)
        
        # Tổ hợp tuyến tính: nhân trọng số và tính tổng dọc theo trục bậc đa thức
        out = (w_view * T_stacked).sum(dim=-1) # Output: [Batch, Seq, Channel]
        
        return out