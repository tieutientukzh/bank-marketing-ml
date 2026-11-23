# Nhận xét EDA – Bank Marketing (bank-additional-full)

Dữ liệu sau khi bỏ 12 dòng trùng: **41.176 dòng**. Script: `notebooks/eda.py` và `notebooks/01_kiem_tra.py`.

## 1. Nhãn mất cân bằng

| y | Số dòng | Tỉ lệ |
|---|---|---|
| no | 36.537 | 88,73% |
| yes | 4.639 | 11,27% |

Một mô hình luôn đoán "no" đã đạt accuracy 88,73%, nên **accuracy không được dùng để chọn mô hình**.

Thay vào đó, nhóm dùng:
- **PR-AUC** làm chỉ số chính;
- F1, recall và precision của lớp yes làm chỉ số phụ.

Khi tách tập phải dùng `stratify=y`. Đã kiểm tra: train và test đều giữ tỉ lệ 0,1127.

## 2. `duration` gây rò rỉ dữ liệu

| y | Trung vị (giây) | Trung bình (giây) | Nhỏ nhất |
|---|---|---|---|
| no | 164 | 220,9 | 0 |
| yes | 449 | 553,3 | 37 |

Hình: `figures/duration_theo_y.png`

- Cuộc gọi của khách "yes" dài gấp khoảng 2,7 lần, tính theo trung vị.
- Cả 4 cuộc gọi có `duration = 0` đều là "no".

`duration` chỉ biết được **sau khi cuộc gọi kết thúc**, tức là lúc đã biết kết quả. Vì vậy:
- **bỏ `duration`** khỏi mô hình dự đoán;
- chỉ dùng nó trong một thí nghiệm đối chứng để chứng minh rò rỉ.

**Thí nghiệm đối chứng** (`notebooks/thi_nghiem_duration.py`, kiểm định chéo 5 phần trên tập train):

| Mô hình | PR-AUC không duration | PR-AUC có duration | ROC-AUC không / có |
|---|---|---|---|
| Hồi quy logistic | 0,4515 | 0,5983 | 0,7898 / 0,9344 |
| Cây quyết định (max_depth=5) | 0,4111 | 0,6095 | 0,7767 / 0,9235 |

Một cột duy nhất làm PR-AUC tăng 0,15–0,20, gấp hơn 10 lần độ lệch giữa các lần chạy (khoảng 0,015). Điểm "có duration" chỉ dùng làm tham chiếu, không dùng để chọn mô hình.
## 3. Giá trị "unknown"

| Cột | Số "unknown" | Tỉ lệ yes của nhóm "unknown" | Tỉ lệ yes của nhóm còn lại |
|---|---|---|---|
| default | 8.596 (20,9%) | 5,2% | no: 12,9% |
| housing | 990 (2,4%) | 10,8% | yes 11,6% / no 10,9% |
| loan | 990 (2,4%) | 10,8% | no 11,3% / yes 10,9% |
| education | 1.730 | 14,5% | 7,8% – 13,7% |
| job | 330 | 11,2% | 6,9% – 31,4% |
| marital | 80 | 15,0% | 10,2% – 14,0% |

- **Giữ "unknown" thành một nhóm riêng**, không điền giá trị khác vào. Ở cột `default`, nhóm "unknown" có tỉ lệ yes chỉ bằng khoảng 40% nhóm "no", tức là bản thân việc "không khai" đã mang thông tin.
- `housing` và `loan` bị "unknown" ở **đúng cùng 990 khách** (kết quả kiểm tra: `True`). Đây là cùng một lỗ hổng dữ liệu, không phải hai lỗi độc lập.
- `default = yes` chỉ có **3 dòng**. Khi tách tập, nhóm này có thể chỉ nằm ở một bên, nên bộ mã hoá phải đặt `OneHotEncoder(handle_unknown="ignore")`.

## 4. `pdays = 999` và `poutcome`

| | failure | nonexistent | success |
|---|---|---|---|
| pdays = 999 | 4.110 | 35.551 | 0 |
| pdays ≠ 999 | 142 | 0 | 1.373 |

- **96,3%** số dòng có `pdays = 999`. Đây là mã đặc biệt, không phải số ngày thật. Nếu để nguyên làm biến số, trung bình bị kéo lên 962 và độ lệch chuẩn là 187, làm hỏng phép chuẩn hoá của hồi quy logistic.
- `nonexistent` luôn đi cùng `pdays = 999`, khớp với mô tả "chưa từng liên hệ".
- Tuy nhiên, **4.110 khách "failure" cũng có `pdays = 999`** (96,7% số failure). Vậy 999 không hoàn toàn đồng nghĩa với "chưa liên hệ". Có lẽ những khách này đã được gọi ở chiến dịch trước nhưng không có ngày liên hệ.
- **Đề xuất:**
  - tạo cờ `da_lien_he_truoc = (pdays != 999)`;
  - không đưa `pdays` thô vào hồi quy logistic;
  - giữ `previous` và `poutcome`.
- `poutcome = success` là tín hiệu mạnh nhất trong các biến phân loại: **65,1% đồng ý**, so với trung bình 11,3%.

## 5. Tỉ lệ yes theo nhóm

Hình: `figures/ti_le_yes_theo_nhom.png`

| Biến | Nhận xét | Mức tín hiệu |
|---|---|---|
| poutcome | success 65,1%; failure 14,2%; nonexistent 8,8% | Rất mạnh |
| month | mar 50,5%, dec 48,9%, sep 44,9%, oct 43,9%. Nhưng 4 tháng này cộng lại chỉ chiếm 4,9% số cuộc gọi. may có 13.767 cuộc gọi (33,4%) nhưng chỉ 6,4% đồng ý | Mạnh, nhưng gắn với thời gian |
| contact | cellular 14,7% so với telephone 5,2% | Mạnh |
| job | student 31,4%, retired 25,3%; blue-collar 6,9% | Mạnh |
| default | no 12,9%, unknown 5,2% | Trung bình |
| education | university 13,7%, basic.9y 7,8%. illiterate 22,2% nhưng chỉ có 18 dòng, không đáng tin | Trung bình |
| marital | single 14,0%, married 10,2% | Yếu–trung bình |
| day_of_week | 10,0%–12,1% | Yếu |
| housing, loan | Chênh nhau dưới 1 điểm % | Gần như không có |

**Kết luận:**
- `housing`, `loan` và `day_of_week` là ứng viên để bỏ trong bước chọn đặc trưng (dòng 6 của bảng thí nghiệm).
- Nhóm quá nhỏ (illiterate 18 dòng, marital unknown 80 dòng, default yes 3 dòng) có tỉ lệ dao động mạnh. Không nên kết luận từ những nhóm này.

## 6. Tuổi có quan hệ hình chữ U

Hình: `figures/tuoi_theo_y.png`

- Khách **dưới 25** và **trên 60** chiếm tỉ trọng cao hơn hẳn ở nhóm yes. Khách 35–55 tuổi chiếm tỉ trọng thấp hơn.
- Kết quả này khớp với tỉ lệ yes theo nghề: student và retired cao nhất.
- Tuổi từ 17 đến 98, trung vị 38.
- **Tương quan Pearson giữa `age` và nhãn chỉ là 0,03**, vì Pearson chỉ đo quan hệ đường thẳng. Một quan hệ hình chữ U cho hệ số gần 0 dù vẫn có thông tin.

**Hệ quả cho hai thuật toán:**
- Hồi quy logistic chỉ học được "tuổi càng cao càng tốt" hoặc "càng xấu", nên cần **thêm đặc trưng nhóm tuổi** (ví dụ <25, 25–60, >60).
- Cây quyết định tự tách được nhiều ngưỡng tuổi, nên không cần.

## 7. Biến số và tương quan

Hình: `figures/tuong_quan.png`

**Tương quan với nhãn:**

| Biến | Hệ số |
|---|---|
| nr.employed | −0,35 |
| pdays | −0,32 (chủ yếu do mã 999) |
| euribor3m | −0,31 |
| emp.var.rate | −0,30 |
| previous | +0,23 |
| cons.price.idx | −0,14 |
| campaign | −0,07 |
| cons.conf.idx | +0,05 |
| age | +0,03 |

**Đa cộng tuyến giữa các chỉ số kinh tế:**
- emp.var.rate – euribor3m: 0,97
- euribor3m – nr.employed: 0,95
- emp.var.rate – nr.employed: 0,91
- emp.var.rate – cons.price.idx: 0,78

Hệ quả:
- Với hồi quy logistic, hệ số của các biến này không ổn định và khó diễn giải. Regularization L2 giúp giảm vấn đề này. Có thể thử chỉ giữ 1–2 biến trong bảng thí nghiệm.
- Với cây quyết định, khả năng dự đoán không bị ảnh hưởng, nhưng độ quan trọng sẽ bị chia cho các biến giống nhau.

**Các chỉ số kinh tế gắn với thời gian.** Số giá trị khác nhau của từng biến: emp.var.rate 10, nr.employed 11, cons.price.idx 26, cons.conf.idx 26, euribor3m 316. Các chỉ số này được đo theo tháng hoặc quý, nên thực chất đóng vai trò "dấu thời gian". Vì vậy, kết quả khi chia ngẫu nhiên có thể lạc quan hơn thực tế. Thí nghiệm chia theo thời gian (nếu kịp) sẽ kiểm tra điều này.

**Biến lệch phải:**
- `campaign`: từ 1 đến 56, trung bình 2,57, 75% số khách ≤ 3.
- `previous`: 75% bằng 0.

Hồi quy logistic nên dùng `log1p(campaign)` hoặc cắt ngưỡng. Cây quyết định không cần.

Thang đo của các biến chênh nhau rất lớn: nr.employed khoảng 5.000, emp.var.rate từ −3,4 đến 1,4. Do đó **bắt buộc dùng StandardScaler cho hồi quy logistic**. Cây quyết định không cần chuẩn hoá.

## 8. Quyết định xử lý dữ liệu

| # | Vấn đề | Quyết định | Áp dụng | Người làm |
|---|---|---|---|---|
| 1 | 12 dòng trùng | Bỏ (`drop_duplicates`) | Cả hai | TV1 – đã xong |
| 2 | `duration` rò rỉ | Bỏ; chỉ dùng trong thí nghiệm đối chứng | Cả hai | TV1 |
| 3 | "unknown" | Giữ thành một nhóm riêng | Cả hai | TV1 |
| 4 | Nhóm hiếm và nhóm lạ trong tập test | `OneHotEncoder(handle_unknown="ignore")` | Hồi quy logistic | TV1 |
| 5 | `pdays = 999` | Cờ `da_lien_he_truoc`; bỏ `pdays` thô | Hồi quy logistic (cây có thể giữ) | TV2 |
| 6 | Tuổi hình chữ U | Thêm nhóm tuổi | Hồi quy logistic | TV2 |
| 7 | `campaign` lệch phải | `log1p` | Hồi quy logistic | TV2 |
| 8 | Thang đo khác nhau | StandardScaler | Hồi quy logistic | TV1 (Pipeline) |
| 9 | Đa cộng tuyến kinh tế | Thử bỏ bớt trong bảng thí nghiệm | Cả hai | TV3 |
| 10 | housing, loan, day_of_week yếu | Thử bỏ ở dòng chọn đặc trưng | Cả hai | TV3 |
| 11 | Mất cân bằng | `stratify`, `class_weight="balanced"`, chỉnh ngưỡng, chọn theo PR-AUC | Cả hai | TV3 |
| 12 | Chỉ số kinh tế và tháng gắn với thời gian | Thí nghiệm chia theo thời gian (nếu kịp) | Cả hai | TV1 |

## 9. Ghi chú về phương pháp

EDA được chạy trên toàn bộ dữ liệu, nhưng **chỉ để mô tả**: không học tham số nào từ dữ liệu này. Mọi bước có học từ dữ liệu đều nằm trong `Pipeline` và chỉ `fit` trên tập train:
- trung bình và độ lệch của StandardScaler;
- danh mục của OneHotEncoder.

Nhờ vậy, tập test không bị rò rỉ vào quá trình huấn luyện.
