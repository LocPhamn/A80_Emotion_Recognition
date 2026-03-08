# 🎥 Webcam Demo Local - Face Tracking + Emotion Detection

Demo chạy webcam **hoàn toàn offline**, không cần API/network. Hữu ích khi demo mà mạng không ổn định.

## ✨ Tính năng

- ✅ Face tracking realtime với ByteTrack
- ✅ Emotion detection với 7 cảm xúc
- ✅ Recording video với thống kê cảm xúc
- ✅ Screenshot
- ✅ Pause/Resume
- ✅ Hoàn toàn offline (không cần backend/AI server)

## 🚀 Cách chạy

### Cách 1: Dùng batch file (Windows)

```bash
run_demo_local.bat
```

### Cách 2: Chạy trực tiếp Python

```bash
cd ai/datn/script
python demo_webcam_local.py
```

## ⌨️ Phím điều khiển

| Phím      | Chức năng              |
| --------- | ---------------------- |
| **R**     | Bắt đầu/Dừng recording |
| **S**     | Chụp screenshot        |
| **Space** | Pause/Resume           |
| **Q**     | Thoát                  |

## 📊 Tính năng Recording

Khi nhấn **R** để bắt đầu recording:

- Video được lưu tự động vào thư mục `demo_recordings/`
- Tracking số lượng người phát hiện được
- Khi dừng recording (nhấn R lại), sẽ hiển thị thống kê cảm xúc đầy đủ:
  - Tổng số người
  - Tỉ lệ phần trăm từng cảm xúc
  - Tổng phải = 100%

### Ví dụ output:

```
⏹️ RECORDING STOPPED
   Session ID: 20260308_143025
   Duration: 45.2s
   Frames: 678
   FPS: 15.0

📊 THỐNG KÊ:
   Tổng số người: 12

   Tỉ lệ cảm xúc (%):
   - Tức giận       :   8.33% (1 người)
   - Chán nản       :   0.00% (0 người)
   - Sợ hãi         :   0.00% (0 người)
   - Hạnh phúc      :  50.00% (6 người)
   - Trung tính     :  41.67% (5 người)
   - Buồn bã        :   0.00% (0 người)
   - Bất ngờ        :   0.00% (0 người)
   ✓ Tổng: 100.00%
```

## 📁 Cấu trúc thư mục output

```
ai/datn/script/
├── demo_webcam_local.py        # Script chính
├── run_demo_local.bat          # Batch file để chạy
├── DEMO_LOCAL_README.md        # File này
└── demo_recordings/            # Thư mục lưu output (tự động tạo)
    ├── demo_20260308_143025.mp4
    ├── demo_20260308_144512.mp4
    └── screenshot_20260308_143156.jpg
```

## 🔧 Cấu hình

Mở file `demo_webcam_local.py` và chỉnh sửa:

```python
# Webcam ID (thay đổi nếu có nhiều webcam)
WEBCAM_ID = 0

# Resolution hiển thị
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# FPS recording
RECORDING_FPS = 15.0
```

## 📋 Requirements

- Python 3.8+
- OpenCV
- PyTorch
- YOLO model
- Emotion classifier model

(Tất cả đã được cài đặt nếu bạn đã chạy được AI server)

## 🆚 So sánh với AI Server

| Tính năng         | AI Server (WebSocket)        | Demo Local        |
| ----------------- | ---------------------------- | ----------------- |
| Network           | ✅ Cần (backend ↔ AI server) | ❌ Không cần      |
| Tracking          | ✅                           | ✅                |
| Emotion Detection | ✅                           | ✅                |
| Recording         | ✅                           | ✅                |
| Statistics        | ✅                           | ✅                |
| Database          | ✅ Lưu vào MySQL             | ❌ Chỉ in console |
| Use case          | Production                   | Demo offline      |

## 💡 Khi nào dùng?

### Dùng Demo Local khi:

- ✅ Demo với khách hàng mà mạng không ổn định
- ✅ Test nhanh model mới
- ✅ Không cần lưu database
- ✅ Chạy standalone

### Dùng AI Server khi:

- ✅ Production environment
- ✅ Cần lưu data vào database
- ✅ Multi-client (nhiều frontend kết nối)
- ✅ Web application

## 🐛 Troubleshooting

### Vấn đề: Webcam không mở được

```
❌ Không thể mở webcam 0
```

**Giải pháp:** Thay đổi `WEBCAM_ID` trong code (thử 1, 2, ...)

### Vấn đề: FPS thấp

**Giải pháp:**

- Giảm resolution: `DISPLAY_WIDTH = 640`, `DISPLAY_HEIGHT = 480`
- Tăng `emotion_cache_frames` trong `FaceEmotionTracker`

### Vấn đề: Model không load được

**Giải pháp:** Kiểm tra đường dẫn model trong `FaceEmotionTracker.__init__()`

## 📝 Notes

- Video được lưu dạng `.mp4` với codec `mp4v`
- Statistics chỉ hiển thị trên console, không lưu database
- Mỗi session recording là độc lập (statistics reset khi start recording mới)
- Frame được xử lý tuần tự (không skip như WebSocket version)

## 📞 Support

Nếu có vấn đề, kiểm tra:

1. Webcam có hoạt động không? (thử ứng dụng Camera của Windows)
2. Model weights có tồn tại không?
3. Python packages đã cài đầy đủ chưa?

---

**Enjoy your demo! 🎉**
