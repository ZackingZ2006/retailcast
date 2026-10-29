# Bắt đầu sử dụng RetailCast

**Mục tiêu:** portfolio Data Science về dự báo số lượng bán trong 28 ngày, có dữ liệu thực, so sánh baseline và đánh giá theo thời gian.

1. Mở `DEMO.html` bằng Chrome/Edge để xem nhanh. File hoạt động offline. Chọn thị trường và cửa sổ xem; xuất CSV khi cần.
2. Đọc `reports/BAO_CAO_KET_QUA.md` để hiểu kết quả và các trường hợp mô hình thất bại.
3. Mở `notebooks/01_research.ipynb` bằng VS Code/Jupyter. Notebook có sẵn kết quả của 11 cell code đã thực thi.
4. Làm theo `README.md` để cài môi trường và tái chạy. Dữ liệu tổng hợp đi kèm đủ để chạy mô hình mà không cần tải lại giao dịch gốc.
5. Đọc `docs/HUONG_DAN_PHONG_VAN.md`, sau đó tự giải thích code và thử một thay đổi có kiểm soát.

## Kết quả cần nhớ đúng
- Holdout: 11/11/2011–08/12/2011, một cửa sổ 28 ngày lịch sử.
- WAPE toàn bộ thị trường: **26,63%**.
- Baseline lặp lại tuần trước: **35,45%**; baseline trung bình theo thứ trong 8 tuần: **28,52%**.
- Mức giảm tương đối so với lặp lại tuần trước: **24,87%**. Đây không phải accuracy và không phải tác động kinh doanh đo được.
- UK dùng baseline trung bình theo thứ vì nó thắng trong validation; không ép ML thắng.
- Champion ở France thua baseline trung bình theo thứ; Denmark thua seasonal naive. Netherlands và Denmark vẫn có WAPE trên 100%.
- Đây là dự báo số hàng đã bán, không phải nhu cầu đã điều chỉnh vì hết hàng. Dự báo tiếp nối sau dữ liệu 2011 không phải dự báo cho năm hiện tại.

## Phần đã kiểm chứng
- 14 kiểm thử dữ liệu và thời gian chạy qua.
- Notebook thực thi đủ 11 cell code và đối chiếu lại một fold GBM với dự báo đã lưu.
- Kiểm toán độc lập đối chiếu hash dữ liệu, ngày chia tập, actual, metric, lựa chọn mô hình và khoảng dự báo.
- Demo HTML kiểm tra logic cho 5 thị trường × 2 chế độ; ứng dụng Streamlit kiểm tra bằng AppTest.
- Các biểu đồ PNG đã được kiểm tra trực quan. Kiểm tra demo trong trình duyệt tích hợp không hoàn tất do kết nối localhost bị timeout; chưa xác nhận giao diện trên mọi kích thước màn hình.

## Khi đưa lên GitHub
Đưa nội dung thư mục `RetailCast` lên repository của bạn. `.gitignore` loại dữ liệu gốc và cache. Giữ notebook, dữ liệu tổng hợp, báo cáo và attribution. Workflow CI đã có sẵn nhưng chỉ chạy sau khi bạn đưa repo lên GitHub; chưa có repository công khai hoặc triển khai Internet nào được tạo.

Bạn cần tự hiểu, chạy lại và phát triển thêm trước khi dùng project trong phỏng vấn. Không nhận là đã triển khai production hoặc tạo ra tiết kiệm tồn kho khi chưa có bằng chứng.
