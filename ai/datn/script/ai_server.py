from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketDisconnect
import cv2
import numpy as np
import base64
import os
import uuid
from pathlib import Path
from fastapi import UploadFile, File, BackgroundTasks
from typing import Optional
import traceback
from datn_ai import FaceEmotionTracker
import datn_ai  # Import module để dùng process_video function

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo model
tracker = FaceEmotionTracker()

# Setup video directories
TEMP_DIR = Path("./temp_videos")
OUTPUT_DIR = Path("./output_videos")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Lưu trạng thái các job đang xử lý
video_jobs = {}

@app.websocket("/ws/process")
async def ai_websocket(ws: WebSocket):
    await ws.accept()
    print("✅ AI WebSocket connected!")
    
    is_processing = False  
    frame_skip_count = 0  
    
    try:
        while True:
            # Nhận frame
            data = await ws.receive_bytes()
            
            # SKIP FRAME nếu đang xử lý
            if is_processing:
                frame_skip_count += 1
                continue
            
            # LOG CẢNH BÁO nếu skip quá nhiều frame
            if frame_skip_count > 5:
                print(f" Bỏ qua {frame_skip_count} frames do xử lý chậm")
            frame_skip_count = 0
                
            is_processing = True
            
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                is_processing = False
                continue

            # Xử lý frame
            result = tracker.process_frame(frame)
            
            # ✅ OPTIMIZATION: Chỉ gửi metadata, không gửi ảnh
            # Giảm bandwidth 100x (từ ~400KB xuống ~2KB)
            await ws.send_json({
                'fps': result['fps'],
                'tracks': result['tracks']
            })
            
            is_processing = False
    
    except WebSocketDisconnect:
        print(" Client không còn kết nối, đóng WebSocket")
    except Exception as e:
        print(f"AI Websocket gặp lỗi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Chỉ close nếu WebSocket chưa đóng
        try:
            await ws.close()
            print(" AI WebSocket đóng")
        except:
            pass

@app.get("/health")
async def health_check():
    """API health kiểm tra endpoint."""
    return {"status": "ok", "service": "AI Emotion Detection"}

# ====== VIDEO PROCESSING ENDPOINTS ======

def process_video_task(job_id: str, input_path: str, output_path: str, skip_frames: int):
    """Background task xử lý video"""
    try:
        # Cập nhật status
        video_jobs[job_id]["status"] = "processing"
        video_jobs[job_id]["progress"] = 0
        
        print(f"🎬 Bắt đầu xử lý video job {job_id}")
        
        # Gọi hàm process_video từ datn_ai với đầy đủ tính năng
        result = datn_ai.FaceEmotionTracker.process_video(
            input_video_path=input_path,
            output_video_path=output_path,
            skip_frames=skip_frames,
            show_preview=False
        )
        
        # Cập nhật kết quả với thông tin đầy đủ
        video_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": {
                "total_visitor": result['total_visitor'],
                "emotion_ratios": result['emotion_ratios'],
                "total_frames": result['total_frames'],
                "processed_frames": result['processed_frames'],
                "fps": result['fps'],
                "resolution": result['resolution']
            }
        })
        
        print(f"✅ Job {job_id} hoàn thành!")
        print(f"   - Total visitors: {result['total_visitor']}")
        print(f"   - Processed frames: {result['processed_frames']}/{result['total_frames']}")
        
    except Exception as e:
        video_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        print(f"❌ Job {job_id} thất bại: {e}")
        traceback.print_exc()

@app.post("/api/video/upload-and-process")
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    skip_frames: int = 1
):
    """Upload video và xử lý với face tracking + emotion detection"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ video format: mp4, avi, mov, mkv")
    
    # Tạo job ID
    job_id = str(uuid.uuid4())
    
    # Tạo đường dẫn file
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    input_path = TEMP_DIR / f"{job_id}_{safe_filename}"
    output_path = OUTPUT_DIR / f"{job_id}_processed.mp4"
    
    # Lưu file upload
    try:
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu file: {str(e)}")
    
    # Tạo job info
    video_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "filename": file.filename,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "error": None
    }
    
    # Chạy background task
    background_tasks.add_task(
        process_video_task,
        job_id,
        str(input_path),
        str(output_path),
        skip_frames
    )
    
    print(f"📤 Job {job_id} đã được tạo cho file: {file.filename}")
    
    return {
        "job_id": job_id,
        "message": "Video đang được xử lý",
        "filename": file.filename,
        "status_url": f"/api/video/status/{job_id}"
    }

@app.get("/api/video/status/{job_id}")
async def get_video_status(job_id: str):
    """Kiểm tra trạng thái xử lý video"""
    
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    
    return video_jobs[job_id]

@app.get("/api/video/download/{job_id}")
async def download_processed_video(job_id: str):
    """Download video đã xử lý"""
    
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    
    job = video_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Video chưa xử lý xong. Status: {job['status']}, Progress: {job.get('progress', 0)}%"
        )
    
    output_path = job["output_path"]
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="File output không tồn tại")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"processed_{job['filename']}"
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)