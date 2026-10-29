# RetailCast — Dự báo doanh số bán lẻ 28 ngày

Dự án dự báo số sản phẩm bán ra mỗi ngày trong 28 ngày cho 5 thị trường, sử dụng dữ liệu giao dịch lịch sử [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). Dự báo hỗ trợ thảo luận lập kế hoạch vận hành.

![So sánh kết quả validation](validation_leaderboard.png)

## Kết quả

Trên một cửa sổ kiểm thử từ 11/11 đến 08/12/2011, phương án được chọn theo từng thị trường đạt WAPE **26,6%**, so với **35,4%** của baseline lặp lại tuần trước và **28,5%** của baseline trung bình theo thứ trong 8 tuần. WAPE tổng hợp bị chi phối bởi thị trường lớn; kết quả tại Denmark và Netherlands vẫn có WAPE trên 100%.

## Phương pháp

- So sánh 6 phương án, gồm baseline, ETS, Ridge và gradient boosting.
- Dùng 4 cửa sổ validation, 2 cửa sổ hiệu chỉnh khoảng và một cửa sổ test cuối; lựa chọn mô hình được khóa trước khi xem test.
- Dùng đặc trưng sẵn có tại thời điểm dự báo, kiểm tra rò rỉ dữ liệu theo thời gian và so sánh từng thị trường.
- Dữ liệu gốc gồm 1.067.371 dòng giao dịch; dữ liệu tổng hợp đi kèm không xuất định danh khách hàng.

## Xem và chạy

- [Notebook phân tích](01_research.ipynb): code và kết quả đã lưu.
- [Demo HTML](DEMO.html): tải về và mở trong trình duyệt để xem tương tác offline.
- [RetailCast_Portfolio.zip](RetailCast_Portfolio.zip): toàn bộ mã nguồn, báo cáo, dữ liệu đã tổng hợp, notebook và hướng dẫn chạy. Giải nén rồi đọc `README.md` và `BAT_DAU.md` trong thư mục.

## Giới hạn và nguồn

Một cửa sổ test 28 ngày không đủ kết luận mô hình ổn định qua mọi mùa. Dữ liệu là lịch sử 2009–2011; dự án không phải dự báo doanh số hiện tại. Số lượng bán ra không đo nhu cầu bị mất do hết hàng, và chưa có thông tin chi phí để khẳng định hiệu quả tồn kho.

Nguồn dữ liệu: Chen, D. (2012), *Online Retail II*, UCI Machine Learning Repository, DOI [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D), CC BY 4.0.
