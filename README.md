Dự đoán khách hàng đăng ký tiền gửi có kỳ hạn – Bank Marketing

Bài cuối kỳ học phần Học máy – Nhóm N01, Khoa Công nghệ Thông tin, Trường Đại học Giao thông Vận tải. Giảng viên hướng dẫn: ThS. Lương Thái Lê.

Bài toán

Một ngân hàng Bồ Đào Nha gọi điện mời khách gửi tiết kiệm có kỳ hạn. Từ 19 thông tin có trước cuộc gọi, mô hình dự đoán xác suất khách đồng ý (y = yes), để ngân hàng ưu tiên gọi những khách có khả năng cao.

Loại bài toán: phân lớp nhị phân. Lớp yes chỉ chiếm 11,27%.
Thuật toán: hồi quy logistic và cây quyết định.
Thước đo chính: PR-AUC. Không chọn mô hình theo accuracy, vì một mô hình luôn đoán "no" đã đạt 88,73%.
Dữ liệu

data/raw/bank-additional-full.csv, bộ Bank Marketing của UCI (Moro, Cortez và Rita, 2014): 41.188 dòng, 20 thuộc tính và nhãn, các cột ngăn cách bằng dấu ;.

Bỏ 12 dòng trùng, còn 41.176 dòng.
Bỏ duration vì gây rò rỉ dữ liệu: độ dài cuộc gọi chỉ biết được sau khi gọi xong. Mô hình còn 19 thuộc tính.
Chia 80% train và 20% test, phân tầng theo y, random_state = 42.
Mọi lựa chọn (đặc trưng, tham số, ngưỡng) đều dựa trên kiểm định chéo phân tầng 5 phần trên tập train. Tập test chỉ dùng một lần cho mô hình cuối.
Cài đặt (Windows PowerShell)
powershell
git clone https://github.com/tieutientukzh/bank-marketing-ml.git
cd bank-marketing-ml
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Nếu PowerShell chặn lệnh Activate.ps1, chạy Set-ExecutionPolicy -Scope CurrentUser RemoteSigned rồi thử lại.

Chạy lại toàn bộ

Mọi lệnh chạy từ thư mục gốc repo, khi môi trường ảo đang bật (dòng lệnh có chữ (.venv) ở đầu).

#	Lệnh	Kết quả
1	python -m notebooks.01_kiem_tra	Kiểm tra nhanh: kích thước, kiểu cột, tỉ lệ nhãn, "unknown", pdays = 999, dòng trùng
2	python -m notebooks.eda	Số liệu EDA và 4 hình trong reports/figures/
3	python -m notebooks.thi_nghiem_duration	Thí nghiệm rò rỉ: reports/thi_nghiem_duration.csv
4	python -m notebooks.bang_thi_nghiem	Bảng thí nghiệm dòng 0–3: reports/bang_thi_nghiem.csv (khoảng 1–2 phút)
5	python -m src.train	Tinh chỉnh tham số, chọn ngưỡng, đánh giá trên test, lưu models/pipeline.joblib và models/threshold.json
6	python -m scripts.smoke_test	Kiểm tra mô hình với 3 khách mẫu và các đường dẫn API
7	python -m uvicorn src.api:app --reload	Mở http://127.0.0.1:8000 để dùng form, hoặc http://127.0.0.1:8000/docs để thử API

Hướng dẫn chi tiết phần API: docs/huong_dan_api_demo.md.

Kết quả thí nghiệm

PR-AUC trung bình qua kiểm định chéo 5 phần trên tập train (reports/bang_thi_nghiem.csv):

Phiên bản	Hồi quy logistic	Cây quyết định (max_depth = 5)
0. Luôn đoán "no"	0,1127	0,1127
1. Dữ liệu thô (không chuẩn hóa)	0,4482 (không hội tụ sau 5000 vòng)	0,4111
2. + Xử lý dữ liệu (chuẩn hóa, sửa mã pdays = 999)	0,4515 (hội tụ sau 70 vòng)	0,4102
2b. Đối chứng: thay "unknown" bằng giá trị phổ biến nhất	0,4510	0,4094
3. + Đặc trưng mới (nhóm tuổi, log1p(campaign))	0,4505	0,4085
Tham chiếu: có duration (rò rỉ)	0,5983	0,6122

Nhận xét chi tiết: reports/nhan_xet_eda.md.

Cấu trúc thư mục
bank-marketing-ml/
├── data/raw/            dữ liệu gốc, không sửa tay
├── notebooks/           script khám phá và thí nghiệm
├── reports/             nhận xét, kết quả CSV, hình
├── src/
│   ├── data.py          đọc dữ liệu, bỏ trùng, bỏ duration, chia train/test
│   ├── features.py      tạo đặc trưng
│   ├── pipeline.py      lắp Pipeline: đặc trưng → tiền xử lý → mô hình
│   ├── evaluate.py      kiểm định chéo, tinh chỉnh, chọn ngưỡng, đánh giá
│   ├── schema.py        kiểu dữ liệu đầu vào của API
│   ├── predict.py       nạp mô hình và dự đoán
│   └── api.py           máy chủ FastAPI
├── scripts/smoke_test.py
├── web/index.html       form nhập thông tin khách hàng
├── docs/                hướng dẫn chạy API
├── models/              mô hình đã huấn luyện
└── requirements.txt
Thành viên
	Họ tên	Phụ trách
TV1	Nguyễn Phúc Hiếu (trưởng nhóm)	Dữ liệu, EDA, thí nghiệm rò rỉ, Pipeline dùng chung, bảng thí nghiệm, tổng hợp
TV2	Lê Thanh Hải	Đặc trưng, schema, API, giao diện web
TV3	TEN_TV3	Tinh chỉnh tham số, xử lý lệch lớp, chọn ngưỡng, đánh giá, mô hình cuối
Nguồn dữ liệu

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22–31. Dữ liệu: UCI Machine Learning Repository, https://doi.org/10.24432/C5K306