# 🏗️ Architecture Overview

## 🚀 Current Setup (Optimized for Performance)

```
┌─────────────┐
│  Frontend   │
│ (React)     │
└──────┬──────┘
       │
       ├──────────────────────────────────┐
       │                                  │
       │ Realtime Webcam                  │ Video Upload
       │ (WebSocket)                      │ (HTTP POST)
       │ Fast Mode                        │ High Accuracy Mode
       │ - 416px resolution               │ - 640px resolution
       │ - Emotion cache 3 frames         │ - Emotion every frame
       │ - ~10-15 FPS                     │ - Full quality
       │                                  │
       ▼                                  ▼
┌──────────────────────────────────────────┐
│          AI Server (Port: 8001)          │
│                                          │
│  • YOLOv11s Face Detection               │
│  • ByteTrack Multi-Object Tracking       │
│  • MobileNetV3 Emotion Classification    │
│                                          │
│  Endpoints:                              │
│  - /ws/process (WebSocket - Realtime)    │
│  - /process_video (HTTP - High Accuracy) │
└──────────────────────────────────────────┘

📝 Backend (Port 8000) không còn được sử dụng
```

## 🎯 Design Decisions

### ✅ Webcam → AI Server (Direct WebSocket)

**Lý do:**

- Giảm 50% latency (bỏ 1 hop qua backend)
- Tăng FPS từ <5 lên ~10-15
- Realtime không cần business logic

**Optimizations:**

- Input size: 416px (balance speed/accuracy)
- Emotion cache: 3 frames
- Frame throttling: 20 FPS max
- Skip frames when processing

**Trade-off:**

- Expose AI server port 8001
- Không có rate limiting/auth (OK cho dev)

### 🎬 Video Upload → AI Server (Direct HTTP)

**Lý do:**

- Không cần realtime → ưu tiên độ chính xác
- Output là video đã visualize (có thể download)
- Xử lý offline, không ảnh hưởng performance

**High Accuracy Mode:**

- Input size: 640px (full resolution)
- Emotion prediction: Every frame (no cache)
- No frame skipping
- Complete video output với tracking + emotion

**Trade-off:**

- Chậm hơn (không cần nhanh)
- File size lớn hơn

## 🔌 Endpoints

### AI Server (FastAPI - Port 8001)

```python
# Realtime WebSocket (Fast Mode)
ws://localhost:8001/ws/process
Input: JPEG frame bytes
Output: JSON {frame: base64, fps: float, tracks: [...]}

# Video Processing (High Accuracy Mode)
POST http://localhost:8001/process_video
Input: multipart/form-data (video file)
Output: JSON {
  success: bool,
  stats: {total_frames, total_faces, emotion_distribution},
  video_base64: string
}

# Download Processed Video
GET http://localhost:8001/download_video/{filename}

# Health Check
GET http://localhost:8001/health
```

### Backend Gateway (Port 8000) - DEPRECATED

```python
# ❌ Không còn sử dụng
# Frontend gọi trực tiếp AI server
```

## 📊 Performance Metrics

| Mode          | Resolution | Emotion Cache | FPS Target | Use Case           |
| ------------- | ---------- | ------------- | ---------- | ------------------ |
| Fast (Webcam) | 416px      | 3 frames      | 10-15      | Realtime detection |
| High (Video)  | 640px      | Every frame   | N/A        | Offline processing |

## 🚦 How to Run

```bash
# Terminal 1: AI Server (REQUIRED - handles both webcam & video)
cd ai/datn/script
python ai_server.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

**Note:** Backend (port 8000) không còn cần thiết!

## 🎬 Features

### 1. Realtime Webcam Detection

- WebSocket connection trực tiếp đến AI server
- Fast mode: 416px, emotion cache 3 frames
- Real FPS tracking ở frontend
- Display: Backend FPS vs Real FPS

### 2. Video Upload Processing

- Upload video file
- High accuracy mode: 640px, emotion every frame
- Progress bar hiển thị upload progress
- Output: Processed video với tracking + emotion visualization
- Download button để lưu video đã xử lý
- Detailed statistics: frames, faces, emotion distribution

## 🔮 Future Improvements

### If needed in production:

1. Add API Gateway (Kong/NGINX) for:

   - Authentication/Authorization
   - Rate limiting
   - Load balancing
   - SSL termination

2. Scale AI Servers:

   - Multiple AI server instances
   - Load balancer
   - Health checks

3. Add monitoring:
   - Prometheus metrics
   - Grafana dashboards
   - Alert system
