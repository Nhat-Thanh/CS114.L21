# **BÀI TẬP QUÁ TRÌNH NGÀY 26/4/2021**

## **Yêu cầu:**
* Mỗi nhóm tìm dăm ba ví dụ về bài toán regression ***TRONG THỰC TẾ***
* Ghi rõ input, output và cách thu thập + xử lý data, commit vào github repository và dẫn link lên Topic trên Classroom.

## **Thực hiện:**
 * 19522246 - Vũ Nguyễn Nhật Thanh
 * 19522180 - Trương Thế Tấn
 * 19521551 - Nông Thanh Hồng

## **Bài toán:**
### **1. Dự đoán số doanh thu quý Highland Coffee dựa vào số lượng sản phẩm bán được.**
* ***input***: Số lượng các mặt hàng nước, đồ ăn bán trong tháng (kiểu số nguyên).
* ***output***: Doanh thu tháng quý tới (kiểu số thực).
* ***Thu thập data***:
    * Tất cả hóa đơn điện tử bán hàng hàng tháng.
    * Khảo sát khách hàng về việc đánh giá và sử dụng sản phầm lần sau.
* ***Xử lý data***:
    * Thống kế số lượng của mỗi sản phẩm.
    * Thống kê lựa chọn sử dụng sản phẩm lần sau của khách hàng
    * Gom chung vào một file SCV

### **2. Dự đoán tình hình dân số Việt Nam dựa vào độ tuổi, tình trạng hôn nhân, số lượng trẻ em được sinh ra và số lượng người chết**
* ***input***: Số lượng dân số ở mỗi độ tuỗi, số lượng nam nữ đã kết hôn ở độ tuổi quy định, số lượng trẻ em được sinh ra và số lượng người chết. (kiểu số nguyên)
* ***Output***: tỷ lệ dân số ở ba nhóm tuổi (số thực)
    * Nhóm tuổi dưới lao động: 0 - 14 tuổi.
    * Nhóm tuổi lao động: 15 - 59 tuổi.
    * Nhóm tuổi trên lao động:  trên 60 tuổi.
* ***Thu thập data***:
    * www.gso.gov.vn :Tổng cục thông kê Việt Nam
* ***Xử lý data***:
    * Tính tỷ suất sinh và tỷ suất tử
    * Thống kế được phần trăm dân số kết hôn
    * Tổng hợp các dữ liệu thành file CSV.

### 3. **Dự đoán tiền lương ngành Công nghệ Thông tin dựa vào bằng cấp, số năm kinh nghiệm, số dự án tham gia**
* ***Input***: Bằng cấp (kiểu chuỗi), số năm kinh nghiệm (kiểu số nguyên), số dự án tham gia (kiểu số nguyên) của các nhân sự trong ngành Công nghệ Thông tin
* ***output***: Lương (kiểu số thực)
* ***Thu nhập data***:
    * Khảo sát các nhân sự trong ngành CNTT
    * Liên hệ với bộ phận nhân sự của các công ty công nghệ như Gameloft, Dek, KMS, Fujinet, VNG, FPT, vv.
* ***Xử lý data***:
    * Gôm nhóm các bằng cấp thành các loại phổ biến.
    * Phân loại giữa bằng cấp ở Việt Nam và nước ngoài.
    * Phân loại dự án tham gia với các vai trò như: 
        * Phân tích và lên kế hoạch
        * Phân tích yêu cầu
        * Thiết kế
        * Phát triển sản phẩm
        * Kiểm thử
        * Triển khai
        * Bảo trì
### 4. **Dự đoán giá nông sản theo tháng trong năm**
* ***Input***: thời gian trong năm theo tháng, độ thuận lợi của thời tiết theo thang 10, sức mua theo thang 10, nhu cầu xuất nhập khẩu, sản lượng thu được theo Kg (Kiểu số nguyên)
* ***output***:Giá bán của nông sản (Kiểu số thực)
* ***Thu thập data***: 
   * Tra cứu trên Google
   * Các trang báo về nông nghiệp
   * Trang web của bộ công thương: www.thongke.idea.gov.vn/
   * Tổng cuc thống kê Việt Nam: www.gso.gov.vn
* ***Xử lý data***:
   * Làm đầy các phần dữ liệu trống(None)
   * Định dạng thành 1 file .csv



