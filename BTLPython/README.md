# Phân loại ảnh thật/giả sử dụng PyTorch

Dự án này sử dụng PyTorch để xây dựng mô hình học sâu phân loại ảnh thật và ảnh do AI tạo ra. Mô hình sử dụng kiến trúc CNN (Convolutional Neural Network) và kỹ thuật Grad-CAM để giải thích quyết định của mô hình.

## Tính năng

- Phân loại ảnh thật/giả với độ chính xác cao
- Giao diện web thân thiện sử dụng Streamlit
- Hiển thị bản đồ nhiệt Grad-CAM để giải thích quyết định
- Hỗ trợ GPU để tăng tốc độ huấn luyện

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
.
├── README.md
├── requirements.txt
├── model.py              # Định nghĩa mô hình CNN
├── data_loader.py        # Xử lý và tải dữ liệu
├── train.py              # Huấn luyện mô hình
└── app.py                # Giao diện web Streamlit
```

## Sử dụng

### 1. Chuẩn bị dữ liệu

Tải bộ dữ liệu CIFAKE từ Kaggle: [https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

Giải nén dữ liệu vào thư mục với cấu trúc:
```
data/
  ├── real/
  │   └── [các ảnh thật]
  └── fake/
      └── [các ảnh giả]
```

### 2. Huấn luyện mô hình

Cập nhật đường dẫn dữ liệu trong file `train.py`:
```python
data_dir = 'path_to_your_data_directory'  # Thay đổi thành đường dẫn thực tế
```

Chạy lệnh huấn luyện:
```bash
python train.py
```

Mô hình sẽ được lưu vào thư mục `saved_model/`.

### 3. Chạy ứng dụng web

```bash
streamlit run app.py
```

Truy cập ứng dụng tại địa chỉ: http://localhost:8501

## Tùy chỉnh

### Tham số huấn luyện

Bạn có thể điều chỉnh các tham số trong file `train.py`:
- `epochs`: Số lần lặp huấn luyện
- `batch_size`: Kích thước batch
- `learning_rate`: Tốc độ học

### Kiến trúc mô hình

Bạn có thể thay đổi kiến trúc mô hình trong file `model.py`:
- Số lượng lớp tích chập
- Số lượng bộ lọc
- Kích thước kernel
- Tỷ lệ dropout

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## Giấy phép

MIT License 