from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from datn_ai import FaceEmotionTracker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo model
tracker = FaceEmotionTracker()

@app.websocket("/ws/process")
async def ai_websocket(ws: WebSocket):
    await ws.accept()
    print("✅ AI WebSocket connected!")
    
    is_processing = False  # ✅ THÊM FLAG
    
    try:
        while True:
            # Nhận frame
            data = await ws.receive_bytes()
            
            # ✅ SKIP FRAME nếu đang xử lý
            if is_processing:
                continue
                
            is_processing = True
            
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                is_processing = False
                continue

            # Xử lý frame
            result = tracker.process_frame(frame)
            
            # ✅ GIẢM CHẤT LƯỢNG JPEG khi encode
            _, buffer = cv2.imencode('.jpg', result['frame'], [
                cv2.IMWRITE_JPEG_QUALITY, 70  # Giảm từ 85 xuống 70
            ])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            await ws.send_json({
                'frame': frame_base64,
                'fps': result['fps'],
                'tracks': result['tracks']
            })
            
            is_processing = False
            
    except Exception as e:
        print(f"❌ AI WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await ws.close()
        print("🔴 AI WebSocket closed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "AI Emotion Detection"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)