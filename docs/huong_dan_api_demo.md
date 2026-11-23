# Hướng dẫn chạy API và demo (phần TV2)

## Chuẩn bị
Mô hình cuối nằm trong `models/` (do TV3 lưu):
- `models/pipeline.joblib`: cả Pipeline (đặc trưng, one-hot, chuẩn hoá, mô hình).
- `models/threshold.json`: ngưỡng quyết định yes/no.

Cài thư viện và chạy từ **thư mục gốc repo**:

```
pip install -r requirements.txt
python -m scripts.smoke_test
python -m uvicorn src.api:app --reload
```

Mở `http://127.0.0.1:8000` để dùng form, hoặc `http://127.0.0.1:8000/docs` để thử API.

## Các endpoint
| Endpoint | Làm gì |
|---|---|
| `GET /` | Trang form nhập khách |
| `GET /health` | Kiểm tra server đang chạy |
| `GET /options` | Danh sách lựa chọn của các biến phân loại |
| `POST /predict` | Nhận 19 thuộc tính của một khách, trả xác suất và kết luận |

## Ví dụ gọi `/predict`
Gửi (không có `duration`):

```json
{
  "age": 40, "job": "admin.", "marital": "married", "education": "university.degree",
  "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
  "month": "may", "day_of_week": "mon", "campaign": 2, "pdays": 999, "previous": 0,
  "poutcome": "nonexistent", "emp_var_rate": 1.1, "cons_price_idx": 93.994,
  "cons_conf_idx": -36.4, "euribor3m": 4.857, "nr_employed": 5191.0
}
```

Nhận:

```json
{"probability": 0.0640, "prediction": "no", "threshold": 0.1941}
```

## Kiểm tra dữ liệu đầu vào
API trả lỗi 422 khi giá trị ngoài phạm vi (ví dụ tuổi 5), nghề không có trong dữ liệu, hoặc các trường mâu thuẫn nhau:
- `poutcome` là `nonexistent` khi và chỉ khi `previous = 0`.
- `previous = 0` thì `pdays` phải là 999.
- `poutcome = success` thì `pdays` phải khác 999.

## Lưu ý
- Phải chạy từ thư mục gốc repo, nếu không file `.joblib` không tìm được module `src`.
- Máy chạy demo cài đúng `requirements.txt`, cùng phiên bản scikit-learn với máy đã lưu mô hình.
- Chưa có `models/pipeline.joblib` thì `/predict` trả lỗi 503.
