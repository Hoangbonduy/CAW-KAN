import pandas as pd

# 1. Đường dẫn file
txt_file_path = '/home/hoang/experiments/CAW-KAN/dataset/individual_household/household_power_consumption.txt'
csv_file_path = '/home/hoang/experiments/CAW-KAN/dataset/individual_household/household_power_consumption.csv'

# 2. Đọc file txt với dấu phân tách là ';'
# Tự động chuyển các ký tự '?' (nếu có) thành giá trị rỗng (NaN) để không lỗi dữ liệu
df = pd.read_csv(txt_file_path, sep=';', low_memory=False, na_values='?')

# 3. Ghép cột 'Date' và 'Time' thành cột mới đặt tên là 'date'
# Kết quả tạm thời sẽ có dạng: "16/12/2006 17:24:00"
df['date'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)

# 4. Chuyển đổi chuỗi vừa ghép sang định dạng datetime chuẩn
# 'format' khai báo cấu trúc gốc trong file txt của bạn (Ngày/Tháng/Năm Giờ:Phút:Giây)
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M:%S')

# 5. Xóa bỏ 2 cột 'Date' và 'Time' cũ sau khi đã gộp thành công
df = df.drop(columns=['Date', 'Time'])

# 6. Đẩy cột 'date' vừa tạo lên vị trí đầu tiên trong bảng
cols = ['date'] + [col for col in df.columns if col != 'date']
df = df[cols]

# 7. Xuất ra file CSV
# Pandas sẽ tự động ghi dữ liệu kiểu datetime dưới định dạng tiêu chuẩn: YYYY-MM-DD HH:MM:SS
df.to_csv(csv_file_path, index=False)

print(f"Đã xử lý và lưu file thành công tại: {csv_file_path}")