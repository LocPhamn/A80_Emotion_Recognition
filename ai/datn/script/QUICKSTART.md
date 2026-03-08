# 🚀 Quick Start - AI Server

## Vấn đề đã sửa:

### 1. WebSocket Disconnect Error ✅

**Lỗi:** `RuntimeError: Cannot call "send" once a close message has been sent.`

**Nguyên nhân:** Khi client đóng WebSocket đột ngột, server cố gửi message hoặc close lại

**Giải pháp:**

- Import `WebSocketDisconnect` từ starlette
- Catch exception riêng cho WebSocketDisconnect
- Wrap `ws.close()` trong try-catch ở finally block

### 2. Video Upload 404 Error ✅

**Lỗi:** `POST /process_video HTTP/1.1" 404 Not Found`

**Nguyên nhân:** Server crash do WebSocket error trước đó

**Giải pháp:**

- Fix WebSocket error trước
- Thêm validation cho video upload
- Thêm error handling và cleanup

---

## 🏃 Chạy Server

```bash
cd ai\datn\script
python ai_server.py
```

Bạn sẽ thấy:

```
🚀 Starting AI Server on port 8001...
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

---

## ✅ Test Server

```bash
# Terminal khác
cd ai\datn\script
python test_server.py
```

Kết quả mong đợi:

```
🔍 Testing AI Server Endpoints...
--------------------------------------------------
✅ Health check: {'status': 'ok', 'service': 'AI Emotion Detection'}
✅ Server is running!

📋 Available endpoints:
  1. WebSocket: ws://localhost:8001/ws/process
  2. POST:      http://localhost:8001/process_video
  3. GET:       http://localhost:8001/download_video/{filename}
  4. GET:       http://localhost:8001/health
--------------------------------------------------
```

---

## 🎯 Sử dụng

### 1. Webcam Realtime (WebSocket)

```javascript
// Frontend
const ws = new WebSocket("ws://localhost:8001/ws/process");

ws.onopen = () => {
  // Gửi frame
  ws.send(frameBlob);
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.frame (base64)
  // data.fps
  // data.tracks
};
```

### 2. Video Upload (HTTP POST)

```javascript
// Frontend
const formData = new FormData();
formData.append("file", videoFile);

const response = await axios.post(
  "http://localhost:8001/process_video",
  formData
);

if (response.data.success) {
  const { stats, video_base64 } = response.data;
  // Display video và stats
}
```

---

## 🐛 Troubleshooting

### Server không start

```bash
# Check port đã được dùng chưa
netstat -ano | findstr :8001

# Kill process nếu cần
taskkill /PID <PID> /F
```

### WebSocket disconnect liên tục

- Check network connection
- Kiểm tra firewall
- Verify frontend URL đúng: `ws://localhost:8001/ws/process`

### Video processing fails

- Check video codec (MP4, AVI, MOV supported)
- Verify disk space cho temp files
- Check console logs cho detailed error

---

## 📊 Logs

Server sẽ log:

- ✅ WebSocket connections
- 📥 Video uploads
- ⚙️ Processing progress
- ❌ Errors với full traceback
- 🗑️ File cleanup

Example:

```
✅ AI WebSocket connected!
⚠️ Skipped 5 frames - processing too slow!
📥 Receiving video: test.mp4
💾 Saved to: temp_videos/abc123_input.mp4 (1048576 bytes)
⚙️ Starting video processing...
🎬 Processing video: temp_videos/abc123_input.mp4
📊 Mode: HIGH ACCURACY
📹 Video info: 1280x720 @ 30fps, 300 frames
✅ High accuracy mode: 640px input, emotion every frame
⏳ Processing frames...
  Progress: 10.0% (30/300)
  Progress: 20.0% (60/300)
  ...
✅ Video saved: output_videos/abc123_output.mp4
📦 Reading output video...
✅ Video processed successfully! Output size: 2097152 bytes
🗑️ Cleaned up temp input file
```

---

## 🔧 Configuration

Trong `ai_server.py`:

```python
# Thư mục temp
TEMP_DIR = Path(__file__).parent / "temp_videos"
OUTPUT_DIR = Path(__file__).parent / "output_videos"

# JPEG quality (70 = balance, 85 = high quality)
cv2.IMWRITE_JPEG_QUALITY, 70

# Port
uvicorn.run(app, host="0.0.0.0", port=8001)
```

Trong `datn_ai.py`:

```python
# Realtime mode (fast)
self.input_size = 416
self.emotion_cache_frames = 3

# High accuracy mode (video processing)
if high_accuracy:
    self.input_size = 640
    self.emotion_cache_frames = 1
```

---

## ✨ Next Steps

1. Start AI server
2. Test với `test_server.py`
3. Start frontend
4. Try webcam realtime
5. Try video upload

Good luck! 🎉
