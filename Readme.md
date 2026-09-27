# Sổ tay từ vựng (Vocabulary SRS)

Ứng dụng học từ vựng Anh–Việt theo phương pháp lặp lại ngắt quãng (Spaced
Repetition), kèm dashboard phân tích dữ liệu học tập.

- `app.py` — ứng dụng chính (Flask): thêm từ, luyện tập, nghe phát âm, SM-2.
- `dashboard.py` — dashboard phân tích & dự đoán (Streamlit).
- `vocab.db` — cơ sở dữ liệu SQLite (từ vựng + lịch sử học).
- `upgrade_db.py` — script migrate schema DB (chạy 1 lần khi cần).

## 1. Yêu cầu

- Python 3.10 trở lên
- Đã cài Git (nếu clone từ repo) hoặc chỉ cần giải nén thư mục project

## 2. Cài đặt (chạy 1 lần)

Mở terminal tại thư mục gốc của project (nơi có file `app.py`):

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Cài toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

## 3. Khởi tạo / cập nhật database (chạy 1 lần, hoặc mỗi khi pull code mới có thay đổi schema)

```bash
python upgrade_db.py
```

Script này sẽ:
- Tạo/vá các cột còn thiếu trong `vocab.db` (an toàn, không mất dữ liệu cũ)
- Tự tải từ điển **NLTK WordNet** (~10MB, chỉ tải lần đầu) để tra loại từ
  (danh từ/động từ/tính từ...) — chạy **hoàn toàn offline**, không phụ
  thuộc API bên ngoài
- Bổ sung loại từ cho các từ đã có sẵn trong sổ

## 4. Chạy ứng dụng

**Ứng dụng học từ vựng (Flask):**
```bash
python app.py
```
Mở trình duyệt tại: http://127.0.0.1:5000

**Dashboard phân tích dữ liệu (Streamlit)** — chạy ở terminal khác:
```bash
streamlit run dashboard.py
```
Streamlit sẽ tự mở trình duyệt tại: http://localhost:8501

## 5. Ghi chú

- Mặc định `app.py` chạy ở chế độ production-safe (`debug=False`). Muốn bật
  chế độ debug khi phát triển:
  ```bash
  set FLASK_DEBUG=1        # Windows (PowerShell: $env:FLASK_DEBUG=1)
  export FLASK_DEBUG=1     # macOS / Linux
  python app.py
  ```
- Tính năng nghe phát âm (🔊) dùng Web Speech API có sẵn trong trình duyệt
  (Chrome/Edge/Safari) — không cần cấu hình gì thêm.
- Nếu thêm từ mới mà phần "Mô tả (Context)" / IPA không hiện ra: ứng dụng
  gọi API `dictionaryapi.dev` để lấy nghĩa/ví dụ — một số mạng (antivirus,
  firewall công ty, một số ISP) có thể chặn ngầm API này. Từ đó vẫn được
  lưu bình thường, chỉ thiếu phần mô tả/ IPA; **loại từ vẫn hiện đúng** vì
  tra hoàn toàn offline (không phụ thuộc API này).