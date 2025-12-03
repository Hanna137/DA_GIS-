ỨNG DỤNG HỌC MÁY MÔ PHỎNG ĐỘ MẶN THEO NGÀY TẠI THÀNH PHỐ HỒ CHÍ MINH

Dự án nhằm mô phỏng và dự báo độ mặn theo ngày tại Thành phố Hồ Chí Minh dựa trên dữ liệu quan trắc thu thập từ 7 trạm đo độ mặn trong giai đoạn 2007–2022. 
Phương pháp chính sử dụng là Random Forest Regression, kết hợp các biến đặc trưng: lưu lượng xả, độ mặn các ngày trước đó, khí tượng, thủy văn. Mục tiêu là làm đầy dữ liệu bị thiếu, mô phỏng độ mặn từng ngày, và đánh giá độ chính xác mô hình bằng R² và RMSE.


DuLieuChung_1Hang1Ngay_SapXepTheoNamff: input đầu vào của mô hình gồm các giá trị về lưu lượng xả, độ mặn, khí tượng, thủy văn
DoMan_Fill_2007_2022_Final1: ouput đầu ra là độ mặn fill theo từng ngày từ năm 2007 - 2020
DanhGia_R2_RMSE_NSEf: độ chính xác của mô hình Random Forest sau khi chạy ra dữ liệu.
FINAL: toàn bộ code để chạy mô hình.
