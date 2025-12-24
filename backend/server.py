from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import websockets
import asyncio
import json

app = FastAPI()

origins = [
    "http://localhost:5174",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_SERVER_URL = "ws://localhost:8001/ws/process"

# ===== UPLOAD VIDEO =====
@app.post("/predict_video")
async def predict_video(file: UploadFile):
    contents = await file.read()
    video_path = "temp.mp4"
    with open(video_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(video_path)
    all_results = []

    # Kết nối đến AI server
    async with websockets.connect(AI_SERVER_URL) as ai_ws:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Gửi đến AI server
            await ai_ws.send(buffer.tobytes())
            
            # Nhận kết quả
            result_json = await ai_ws.recv()
            result = json.loads(result_json)
            all_results.append(result)

    cap.release()
    return {"frames": len(all_results), "results": all_results}


# ===== WEBSOCKET REALTIME - PROXY TO AI SERVER =====
@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    print("✅ Client WebSocket connected!")
    
    try:
        # Kết nối đến AI server
        async with websockets.connect(AI_SERVER_URL) as ai_ws:
            print("✅ Connected to AI Server!")
            
            async def forward_to_ai():
                """Nhận frame từ frontend -> gửi đến AI server"""
                try:
                    while True:
                        data = await client_ws.receive_bytes()
                        await ai_ws.send(data)
                except WebSocketDisconnect:
                    print("🔴 Client disconnected")
                except Exception as e:
                    print(f"❌ Forward error: {e}")
            
            async def forward_to_client():
                """Nhận kết quả từ AI server -> gửi đến frontend"""
                try:
                    while True:
                        result = await ai_ws.recv()
                        await client_ws.send_text(result)
                except Exception as e:
                    print(f"❌ Receive error: {e}")
            
            # Chạy song song 2 task
            await asyncio.gather(
                forward_to_ai(),
                forward_to_client()
            )
            
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        await client_ws.close()
        print("🔴 WebSocket closed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with websockets.connect(AI_SERVER_URL) as ws:
            return {"status": "ok", "ai_server": "connected"}
    except:
        return {"status": "error", "ai_server": "disconnected"}


@app.get("/")
async def root():
    return {"message": "Backend Server - Gateway to AI Service"}