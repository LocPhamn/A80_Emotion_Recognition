from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import websockets
import asyncio
import json
import aiohttp
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from typing import Optional, List
import logging
import requests
import os
from pathlib import Path
import uuid
import mimetypes
import subprocess
import shutil
from collections import defaultdict
import time

# Import models
from models import VideoResponse, VideoCreate, VideoUpdate, StatisticResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set console handler encoding to utf-8
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.stream.reconfigure(encoding='utf-8') if hasattr(handler.stream, 'reconfigure') else None

app = FastAPI()

# MySQL Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Locpro@1997',
    'database': 'video_management',
    'port': 3306
}

# Database helper functions
def get_db_connection():
    """Create database connection"""
    try:
        logger.info(f"Attempting to connect to MySQL database: {DB_CONFIG['database']} at {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        connection = mysql.connector.connect(**DB_CONFIG)
        logger.info("✅ Database connection established successfully")
        return connection
    except Error as e:
        logger.error(f"❌ Error connecting to MySQL: {e}")
        logger.error(f"Connection details: host={DB_CONFIG['host']}, user={DB_CONFIG['user']}, database={DB_CONFIG['database']}, port={DB_CONFIG['port']}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def close_db_connection(connection, cursor=None):
    """Close database connection"""
    try:
        if cursor:
            cursor.close()
            logger.debug("Database cursor closed")
        if connection and connection.is_connected():
            connection.close()
            logger.info("✅ Database connection closed successfully")
    except Error as e:
        logger.error(f"❌ Error closing database connection: {e}")

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
AI_SERVER_HTTP = "http://localhost:8001"

# Video storage directory
VIDEO_STORAGE_DIR = Path("./uploaded_videos")
VIDEO_STORAGE_DIR.mkdir(exist_ok=True)

# Store processing jobs
video_processing_jobs = {}

# Store recording sessions
recording_sessions = {}


def convert_to_web_compatible(input_path: Path, output_path: Path) -> bool:
    """Convert video to H.264 + AAC for web browser compatibility"""
    try:
        logger.info(f"[CONVERT] Converting video to web-compatible format: {input_path.name}")
        
        # Check if ffmpeg is available
        if not shutil.which('ffmpeg'):
            logger.warning("[CONVERT] ffmpeg not found, skipping conversion")
            return False
        
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',      # H.264 video codec
            '-preset', 'fast',      # Encoding speed
            '-crf', '23',           # Quality (lower = better, 18-28 recommended)
            '-c:a', 'aac',          # AAC audio codec
            '-b:a', '128k',         # Audio bitrate
            '-movflags', '+faststart',  # Enable progressive streaming
            '-y',                   # Overwrite output file
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            logger.info(f"[CONVERT] Video converted successfully: {output_path.name}")
            return True
        else:
            logger.error(f"[CONVERT] ffmpeg conversion failed: {result.stderr.decode()}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("[CONVERT] Video conversion timeout (>5 minutes)")
        return False
    except Exception as e:
        logger.error(f"[CONVERT] Error converting video: {e}")
        return False

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
        print(f"WebSocket error: {e}")
    finally:
        await client_ws.close()
        print("WebSocket closed")


# ===== WEBSOCKET WITH RECORDING =====
@app.websocket("/ws/process")
async def process_webcam_with_recording(websocket: WebSocket):
    """
    WebSocket endpoint để xử lý webcam realtime và ghi video
    Client sẽ gửi:
    - Frames (binary) để xử lý
    - Control messages (JSON) để bắt đầu/dừng recording
    """
    await websocket.accept()
    logger.info("✅ Webcam WebSocket connected")
    
    session_id = str(uuid.uuid4())
    recording_sessions[session_id] = {
        "is_recording": False,
        "video_writer": None,
        "raw_video_path": None,
        "start_time": None,
        "frame_count": 0,
        "fps": 15,
        "width": None,
        "height": None
    }
    
    ai_ws = None
    
    try:
        # Kết nối đến AI server
        ai_ws = await websockets.connect(AI_SERVER_URL)
        logger.info("✅ Connected to AI Server for webcam processing")
        
        async def receive_from_client():
            """Nhận frames/commands từ client"""
            session = recording_sessions[session_id]
            
            try:
                while True:
                    # Nhận data từ client (có thể là frame hoặc command)
                    try:
                        data = await websocket.receive()
                        
                        # Kiểm tra nếu là text command
                        if 'text' in data:
                            command = json.loads(data['text'])
                            
                            if command['type'] == 'start_recording':
                                # Bắt đầu recording
                                zone_id = command.get('zone_id', 1)
                                session['zone_id'] = zone_id
                                session['is_recording'] = True
                                session['start_time'] = time.time()
                                session['frame_count'] = 0
                                
                                # Tạo video file TẠM (raw webcam frames)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                raw_filename = f"webcam_raw_{timestamp}.mp4"
                                session['raw_video_path'] = VIDEO_STORAGE_DIR / raw_filename
                                
                                logger.info(f"🔴 Started recording: {raw_filename}")
                                
                                await websocket.send_text(json.dumps({
                                    "type": "recording_started",
                                    "session_id": session_id,
                                    "filename": raw_filename
                                }))
                                
                            elif command['type'] == 'stop_recording':
                                # Dừng recording và xử lý video
                                if session['is_recording']:
                                    await finalize_recording(session_id, websocket)
                                
                        # Nếu là binary data (frame)
                        elif 'bytes' in data:
                            frame_bytes = data['bytes']
                            
                            # Gửi frame đến AI server để xử lý real-time
                            await ai_ws.send(frame_bytes)
                            
                            # Nếu đang recording, ghi frame vào video raw
                            if session['is_recording']:
                                nparr = np.frombuffer(frame_bytes, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if frame is not None:
                                    # Initialize video writer nếu chưa có
                                    if session['video_writer'] is None:
                                        height, width = frame.shape[:2]
                                        session['width'] = width
                                        session['height'] = height
                                        
                                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                        session['video_writer'] = cv2.VideoWriter(
                                            str(session['raw_video_path']),
                                            fourcc,
                                            15.0,  # 15 FPS
                                            (width, height)
                                        )
                                        logger.info(f"📹 Video writer initialized: {width}x{height} @ 15 FPS")
                                    
                                    # Ghi frame vào video
                                    if session['video_writer']:
                                        session['video_writer'].write(frame)
                                        session['frame_count'] += 1
                                        if session['frame_count'] % 50 == 0:
                                            logger.info(f"📹 Recorded {session['frame_count']} frames")
                            
                    except WebSocketDisconnect:
                        logger.info("🔴 Client disconnected")
                        break
                    except Exception as e:
                        logger.error(f"❌ Error receiving from client: {e}")
                        import traceback
                        traceback.print_exc()
                        break
                        
            except Exception as e:
                logger.error(f"❌ Error in receive_from_client: {e}")
                import traceback
                traceback.print_exc()
        
        async def send_to_client():
            """Nhận kết quả từ AI server và gửi về client"""
            session = recording_sessions[session_id]
            
            try:
                while True:
                    # Nhận kết quả từ AI server
                    result_json = await ai_ws.recv()
                    result = json.loads(result_json)
                    
                    # Gửi kết quả về client
                    await websocket.send_text(result_json)
                    
            except Exception as e:
                logger.error(f"❌ Error in send_to_client: {e}")
                import traceback
                traceback.print_exc()
        
        # Chạy song song 2 tasks
        await asyncio.gather(
            receive_from_client(),
            send_to_client()
        )
        
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        # Cleanup
        if recording_sessions[session_id]['is_recording']:
            await finalize_recording(session_id, websocket)
        
        if ai_ws:
            await ai_ws.close()
        
        await websocket.close()
        del recording_sessions[session_id]
        logger.info("🔴 WebSocket closed and cleaned up")


async def safe_send_websocket(websocket: WebSocket, message: dict):
    """Safely send message to WebSocket, ignore if connection is closed"""
    try:
        # Check if WebSocket is still open
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_text(json.dumps(message))
            logger.info(f"✅ Sent message to client: {message.get('type')}")
        else:
            logger.warning(f"⚠️ WebSocket already closed, cannot send: {message.get('type')}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to send WebSocket message: {e}")

async def finalize_recording(session_id: str, websocket: WebSocket):
    """Hoàn tất recording, gọi AI server để xử lý video, và lưu vào database"""
    session = recording_sessions[session_id]
    
    try:
        logger.info(f"⏹️ Stopping recording: {session.get('raw_video_path')}")
        
        # Đóng video writer
        if session['video_writer']:
            session['video_writer'].release()
            session['video_writer'] = None
            logger.info("✅ Video writer closed")
        
        raw_video_path = session.get('raw_video_path')
        if not raw_video_path or not raw_video_path.exists():
            raise Exception("Raw video file not found")
        
        logger.info(f"📤 Uploading video to AI Server for processing...")
        
        # Upload video lên AI server để xử lý
        try:
            async with aiohttp.ClientSession() as http_session:
                with open(raw_video_path, 'rb') as f:
                    form = aiohttp.FormData()
                    form.add_field('file', f, filename=raw_video_path.name)
                    form.add_field('skip_frames', '1')
                    
                    async with http_session.post(
                        f"{AI_SERVER_HTTP}/api/video/upload-and-process",
                        data=form
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"AI server returned {response.status}")
                        
                        result = await response.json()
                        ai_job_id = result['job_id']
                        logger.info(f"✅ AI Job created: {ai_job_id}")
        except Exception as e:
            logger.error(f"❌ Failed to upload to AI server: {e}")
            raise
        
        # Poll AI server cho đến khi hoàn thành
        logger.info(f"⏳ Waiting for AI processing to complete...")
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        status_data = None
        while True:
            if time.time() - start_time > max_wait_time:
                raise Exception("AI processing timeout")
            
            try:
                async with aiohttp.ClientSession() as http_session:
                    async with http_session.get(
                        f"{AI_SERVER_HTTP}/api/video/status/{ai_job_id}"
                    ) as response:
                        status_data = await response.json()
                        
                        logger.info(f"📊 AI Status: {status_data.get('status')} - Progress: {status_data.get('progress', 0)}%")
                        
                        if status_data['status'] == 'completed':
                            logger.info("✅ AI processing completed!")
                            logger.info(f"📊 Result: {status_data.get('result')}")
                            break
                        elif status_data['status'] == 'failed':
                            raise Exception(f"AI processing failed: {status_data.get('error')}")
                        
                        # Đợi 2 giây trước khi check lại
                        await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"❌ Error checking AI status: {e}")
                raise
        
        # Lấy statistics từ AI result
        if not status_data or 'result' not in status_data:
            logger.error(f"❌ No result in status_data: {status_data}")
            raise Exception("AI server did not return result")
            
        ai_result = status_data.get('result', {})
        total_visitor = ai_result.get('total_visitor', 0)
        emotion_ratios = ai_result.get('emotion_ratios', {})
        
        logger.info(f"📊 Statistics from AI: visitors={total_visitor}, emotions={emotion_ratios}")
        
        # Download processed video từ AI server
        processed_video_path = VIDEO_STORAGE_DIR / f"webcam_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.get(
                    f"{AI_SERVER_HTTP}/api/video/download/{ai_job_id}"
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Download failed: {response.status}")
                    
                    with open(processed_video_path, 'wb') as f:
                        f.write(await response.read())
                    
                    logger.info(f"✅ Processed video downloaded: {processed_video_path.name}")
        except Exception as e:
            logger.error(f"❌ Failed to download processed video: {e}")
            raise
        
        # Tính duration
        duration = int(time.time() - session['start_time']) if session['start_time'] else 0
        
        # Convert video to web-compatible format
        final_video_path = processed_video_path.parent / f"{processed_video_path.stem}_web{processed_video_path.suffix}"
        if convert_to_web_compatible(processed_video_path, final_video_path):
            processed_video_path.unlink()  # Xóa file gốc
            logger.info(f"✅ Video converted to web format: {final_video_path.name}")
        else:
            final_video_path = processed_video_path
            logger.warning("⚠️ Video conversion failed, using original format")
        
        # Xóa raw video
        if raw_video_path.exists():
            raw_video_path.unlink()
            logger.info("🗑️ Raw video deleted")
        
        # Lưu vào database
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        try:
            # Tạo video record
            video_query = """
                INSERT INTO video (video_name, zone_id, duration, date, file_path, status)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            video_name = final_video_path.name
            cursor.execute(video_query, (
                video_name,
                session.get('zone_id', 1),
                duration,
                datetime.now(),
                str(final_video_path),
                'process'
            ))
            connection.commit()
            
            video_id = cursor.lastrowid
            logger.info(f"✅ Video saved to DB: ID={video_id}, Name={video_name}")
            
            # Tạo statistic record
            stat_query = """
                INSERT INTO statistic 
                (video_id, total_visitor, angry_rate, disgust_rate, fear_rate, 
                 happy_rate, neutral_rate, sad_rate, surprise_rate)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            stat_values = (
                video_id,
                total_visitor,
                emotion_ratios.get('angry', 0),
                emotion_ratios.get('disgust', 0),
                emotion_ratios.get('fear', 0),
                emotion_ratios.get('happy', 0),
                emotion_ratios.get('neutral', 0),
                emotion_ratios.get('sad', 0),
                emotion_ratios.get('surprise', 0)
            )
            
            logger.info(f"📊 Inserting statistics: {stat_values}")
            
            cursor.execute(stat_query, stat_values)
            connection.commit()
            
            logger.info(f"✅ Statistics saved for video {video_id}: visitors={total_visitor}, emotions={emotion_ratios}")
            
            # Gửi thông báo hoàn tất về client (nếu connection còn mở)
            await safe_send_websocket(websocket, {
                "type": "recording_stopped",
                "video_id": video_id,
                "filename": video_name,
                "duration": duration,
                "frame_count": session['frame_count'],
                "total_visitor": total_visitor,
                "statistics": {
                    "total_visitor": total_visitor,
                    "emotion_rates": emotion_ratios
                }
            })
            
            session['is_recording'] = False
            
        except Error as e:
            connection.rollback()
            logger.error(f"❌ Database error while saving recording: {e}")
            raise
        finally:
            close_db_connection(connection, cursor)
            
    except Exception as e:
        logger.error(f"❌ Error finalizing recording: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to send error to client (nếu connection còn mở)
        await safe_send_websocket(websocket, {
            "type": "recording_error",
            "error": str(e)
        })


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with websockets.connect(AI_SERVER_URL) as ws:
            return {"status": "ok", "ai_server": "connected"}
    except:
        return {"status": "error", "ai_server": "disconnected"}


@app.get("/api/db-check")
async def check_database():
    """Check database connection"""
    logger.info("🔍 Database connection check requested")
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Check connection
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        
        # Check tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM video")
        video_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM statistic")
        stat_count = cursor.fetchone()[0]
        
        close_db_connection(connection, cursor)
        
        logger.info(f"✅ Database check successful - {video_count} videos, {stat_count} statistics")
        
        return {
            "status": "connected",
            "database": DB_CONFIG['database'],
            "mysql_version": version[0],
            "tables": table_names,
            "video_count": video_count,
            "statistic_count": stat_count
        }
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/")
async def root():
    return {"message": "Backend Server - Gateway to AI Service"}
os.makedirs('video_storage', exist_ok=True)

# ===== VIDEO UPLOAD & PROCESS WITH DATABASE =====
@app.post("/api/video/upload-process")
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    zone_id: int = Form(...)
):
    """Upload video, process với AI, và lưu vào database"""
    
    logger.info(f"📤 Upload video: {file.filename}, Zone ID: {zone_id}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ video format: mp4, avi, mov, mkv, webm")
    
    try:
        # Tạo job ID
        job_id = str(uuid.uuid4())
        
        # Upload video lên AI server
        files = {'file': (file.filename, await file.read(), file.content_type)}
        response = requests.post(
            f"{AI_SERVER_HTTP}/api/video/upload-and-process",
            files=files
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Không thể upload lên AI server")
        
        ai_job_id = response.json()['job_id']
        
        # Lưu job info
        video_processing_jobs[job_id] = {
            "ai_job_id": ai_job_id,
            "filename": file.filename,
            "zone_id": zone_id,
            "status": "processing",
            "progress": 0
        }
        
        # Background task để poll AI server và lưu vào DB
        background_tasks.add_task(
            poll_and_save_video,
            job_id,
            ai_job_id,
            file.filename,
            zone_id
        )
        
        logger.info(f"✅ Job {job_id} created, AI Job: {ai_job_id}")
        
        return {
            "job_id": job_id,
            "message": "Video đang được xử lý",
            "filename": file.filename,
            "status_url": f"/api/video/process-status/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/webcam/save-recording")
async def save_webcam_recording(
    session_id: str = Form(...),
    video_path: str = Form(...),
    duration: float = Form(...),
    statistics: str = Form(...),
    zone_id: int = Form(1)
):
    """
    Lưu webcam recording từ AI Server vào database
    
    Args:
        session_id: Session ID từ AI server
        video_path: Đường dẫn video trên AI server
        duration: Độ dài video (seconds)
        statistics: JSON string chứa statistics
        zone_id: Zone ID (default=1)
    
    Returns:
        Video ID và statistics
    """
    logger.info(f"📤 Saving webcam recording: session={session_id}, zone={zone_id}")
    
    try:
        # Parse statistics
        stats = json.loads(statistics)
        logger.info(f"📊 Statistics: {stats}")
        
        # Download video từ AI server
        ai_download_url = f"{AI_SERVER_HTTP}/api/webcam/download/{session_id}"
        response = requests.get(ai_download_url, stream=True)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Cannot download video from AI server")
        
        # Lưu video vào storage
        video_filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_storage_path = VIDEO_STORAGE_DIR / video_filename
        
        with open(video_storage_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"💾 Video saved to: {video_storage_path}")
        
        # Convert to web-compatible format
        converted_path = video_storage_path.parent / f"{video_storage_path.stem}_web{video_storage_path.suffix}"
        if convert_to_web_compatible(video_storage_path, converted_path):
            final_video_path = converted_path
            try:
                video_storage_path.unlink()
                logger.info(f"[CLEANUP] Deleted original video")
            except:
                pass
        else:
            final_video_path = video_storage_path
        
        # Lưu vào database
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        try:
            # Tạo video record
            video_query = """
                INSERT INTO video (video_name, zone_id, duration, date, file_path, status)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(video_query, (
                video_filename,
                zone_id,
                int(duration),
                datetime.now(),
                str(final_video_path),
                'process'
            ))
            connection.commit()
            
            video_id = cursor.lastrowid
            logger.info(f"✅ Video record created: ID={video_id}")
            
            # Tạo statistic record
            emotion_ratios = stats.get('emotion_ratios', {})
            
            # Map emotion names to indices
            emotion_map = {
                'angry': '0',
                'disgust': '1',
                'fear': '2',
                'happy': '3',
                'neutral': '4',
                'sad': '5',
                'surprise': '6'
            }
            
            stat_query = """
                INSERT INTO statistic 
                (video_id, total_visitor, angry_rate, disgust_rate, fear_rate, 
                 happy_rate, neutral_rate, sad_rate, surprise_rate)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(stat_query, (
                video_id,
                stats.get('total_visitor', 0),
                emotion_ratios.get('angry', {}).get('ratio', 0),
                emotion_ratios.get('disgust', {}).get('ratio', 0),
                emotion_ratios.get('fear', {}).get('ratio', 0),
                emotion_ratios.get('happy', {}).get('ratio', 0),
                emotion_ratios.get('neutral', {}).get('ratio', 0),
                emotion_ratios.get('sad', {}).get('ratio', 0),
                emotion_ratios.get('surprise', {}).get('ratio', 0)
            ))
            connection.commit()
            
            logger.info(f"✅ Statistic record created for video {video_id}")
            
            return {
                "video_id": video_id,
                "video_name": video_filename,
                "duration": duration,
                "statistics": stats
            }
            
        except Error as e:
            connection.rollback()
            logger.error(f"❌ Database error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            close_db_connection(connection, cursor)
            
    except Exception as e:
        logger.error(f"❌ Save recording error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/webcam/upload-recording")
async def upload_webcam_recording(
    video: UploadFile = File(...),
    emotion_log: str = Form(...),
    zone_id: Optional[int] = Form(1),
    duration: Optional[int] = Form(None)
):
    """
    Upload webcam recording (.webm) với emotion log từ frontend
    
    Args:
        video: File .webm từ MediaRecorder
        emotion_log: JSON string chứa emotion data cho từng frame
        zone_id: ID của zone (default=1)
        duration: Độ dài video (seconds)
    
    Returns:
        {
            "video_id": int,
            "video_name": str,
            "statistics": {...}
        }
    """
    
    logger.info(f"📤 Webcam recording upload: {video.filename}, Zone ID: {zone_id}")
    
    try:
        # Parse emotion log
        emotion_data = json.loads(emotion_log)
        logger.info(f"📊 Emotion log entries: {len(emotion_data)}")
        
        # Create temp directories
        temp_dir = Path("temp_webcam")
        temp_dir.mkdir(exist_ok=True)
        
        # Save .webm file
        webm_path = temp_dir / f"webcam_{uuid.uuid4()}.webm"
        with open(webm_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        logger.info(f"💾 Saved .webm to {webm_path}")
        
        # Convert .webm to .mp4 using ffmpeg
        mp4_filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        mp4_path = Path("video_storage") / mp4_filename
        mp4_path.parent.mkdir(exist_ok=True)
        
        # FFmpeg command: convert webm to mp4 with H.264
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(webm_path),
            "-c:v", "libx264",  # H.264 codec
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-y",  # Overwrite
            str(mp4_path)
        ]
        
        logger.info(f"🎬 Converting .webm to .mp4...")
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            logger.error(f"❌ FFmpeg error: {result.stderr}")
            raise HTTPException(status_code=500, detail="Video conversion failed")
        
        logger.info(f"✅ Converted to {mp4_path}")
        
        # Get video metadata
        cap = cv2.VideoCapture(str(mp4_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if duration is None:
            duration = int(total_frames / fps) if fps > 0 else 0
        cap.release()
        
        # Calculate statistics from emotion log
        statistics = calculate_statistics_from_log(emotion_data, duration)
        
        logger.info(f"📊 Statistics: {statistics}")
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Insert video with all required fields and default values
            video_query = """
                INSERT INTO video (video_name, zone_id, duration, date, file_path, status)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(video_query, (mp4_filename, zone_id, duration, current_date, str(mp4_path), 'process'))
            video_id = cursor.lastrowid
            
            # Insert statistics
            statistic_query = """
                INSERT INTO statistic (
                    video_id, total_visitor, happy_rate, sad_rate, 
                    angry_rate, neutral_rate, surprise_rate, 
                    fear_rate, disgust_rate
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(statistic_query, (
                video_id,
                statistics['total_visitor'],
                statistics['happy_rate'],
                statistics['sad_rate'],
                statistics['angry_rate'],
                statistics['neutral_rate'],
                statistics['surprise_rate'],
                statistics['fear_rate'],
                statistics['disgust_rate']
            ))
            
            conn.commit()
            
            logger.info(f"✅ Saved video ID {video_id} and statistics to database")
            
            # Clean up temp file
            webm_path.unlink()
            
            return {
                "video_id": video_id,
                "video_name": mp4_filename,
                "statistics": statistics,
                "message": "Webcam recording saved successfully"
            }
            
        finally:
            cursor.close()
            conn.close()
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid emotion log JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid emotion log format")
    except Exception as e:
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def calculate_statistics_from_log(emotion_data: list, duration: int) -> dict:
    """
    Tính statistics từ emotion log
    
    Args:
        emotion_data: List of {frame, track_id, emotion, confidence, bbox, timestamp}
        duration: Video duration in seconds
    
    Returns:
        {
            'total_visitor': int,
            'happy_rate': float,
            'sad_rate': float,
            ...
        }
    """
    
    # Count unique track IDs
    unique_tracks = set()
    emotion_counts = defaultdict(int)
    
    for entry in emotion_data:
        track_id = entry.get('track_id')
        emotion = entry.get('emotion', 'neutral')
        
        if track_id is not None:
            unique_tracks.add(track_id)
        
        # Count emotions
        emotion_counts[emotion.lower()] += 1
    
    total_visitor = len(unique_tracks)
    total_emotions = sum(emotion_counts.values())
    
    # Calculate emotion rates (%)
    emotion_rates = {}
    for emotion in ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']:
        count = emotion_counts.get(emotion, 0)
        rate = (count / total_emotions * 100) if total_emotions > 0 else 0.0
        emotion_rates[f'{emotion}_rate'] = round(rate, 2)
    
    logger.info(f"📊 Statistics: {total_visitor} visitors, {total_emotions} emotion detections")
    
    return {
        'total_visitor': total_visitor,
        **emotion_rates
    }


async def poll_and_save_video(job_id: str, ai_job_id: str, filename: str, zone_id: int):
    """Poll AI server status và lưu vào database khi hoàn thành"""
    
    try:
        # Poll AI server
        while True:
            await asyncio.sleep(2)  # Poll mỗi 2 giây
            
            response = requests.get(f"{AI_SERVER_HTTP}/api/video/status/{ai_job_id}")
            data = response.json()
            
            # Cập nhật progress
            video_processing_jobs[job_id]["progress"] = data.get("progress", 0)
            video_processing_jobs[job_id]["status"] = data["status"]
            
            if data["status"] == "completed":
                logger.info(f"✅ AI processing completed for job {job_id}")
                
                # Lấy kết quả
                result = data["result"]
                
                # Download processed video
                download_response = requests.get(
                    f"{AI_SERVER_HTTP}/api/video/download/{ai_job_id}",
                    stream=True
                )
                
                # Lưu video vào storage
                video_name = Path(filename).stem
                output_filename = f"{video_name}_processed_{job_id}.mp4"
                output_path = VIDEO_STORAGE_DIR / output_filename
                
                with open(output_path, 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"[UPLOAD] Video saved: {output_path}")
                
                # Convert video to web-compatible format (H.264 + AAC)
                converted_path = output_path.parent / f"{output_path.stem}_web{output_path.suffix}"
                if convert_to_web_compatible(output_path, converted_path):
                    # Use converted video
                    final_video_path = converted_path
                    # Delete original non-compatible video
                    try:
                        output_path.unlink()
                        logger.info(f"[CLEANUP] Deleted original video: {output_path.name}")
                    except Exception as e:
                        logger.warning(f"[CLEANUP] Could not delete original video: {e}")
                else:
                    # Use original video if conversion failed
                    logger.warning("[CONVERT] Using original video (conversion failed or ffmpeg not available)")
                    final_video_path = output_path
                
                # Tính duration từ total_frames và fps
                duration = int(result['total_frames'] / result['fps']) if result['fps'] > 0 else 0
                
                # Lưu vào database
                connection = get_db_connection()
                cursor = connection.cursor(dictionary=True)
                
                try:
                    # Tạo video record
                    video_query = """
                        INSERT INTO video (video_name, zone_id, duration, date, file_path, status)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(video_query, (
                        filename,
                        zone_id,
                        duration,
                        datetime.now(),
                        str(final_video_path),
                        'process'
                    ))
                    connection.commit()
                    
                    video_id = cursor.lastrowid
                    logger.info(f"✅ Video record created: ID={video_id}")
                    
                    # Tạo statistic record
                    emotion_ratios = result['emotion_ratios']
                    
                    stat_query = """
                        INSERT INTO statistic 
                        (video_id, total_visitor, angry_rate, disgust_rate, fear_rate, 
                         happy_rate, neutral_rate, sad_rate, surprise_rate)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(stat_query, (
                        video_id,
                        result['total_visitor'],
                        emotion_ratios.get('0', {}).get('ratio', 0),  # angry
                        emotion_ratios.get('1', {}).get('ratio', 0),  # disgust
                        emotion_ratios.get('2', {}).get('ratio', 0),  # fear
                        emotion_ratios.get('3', {}).get('ratio', 0),  # happy
                        emotion_ratios.get('4', {}).get('ratio', 0),  # neutral
                        emotion_ratios.get('5', {}).get('ratio', 0),  # sad
                        emotion_ratios.get('6', {}).get('ratio', 0)   # surprise
                    ))
                    connection.commit()
                    
                    logger.info(f"✅ Statistic record created for video {video_id}")
                    
                    # Cập nhật job status
                    video_processing_jobs[job_id].update({
                        "status": "completed",
                        "video_id": video_id,
                        "result": result
                    })
                    
                except Error as e:
                    connection.rollback()
                    logger.error(f"❌ Database error: {e}")
                    video_processing_jobs[job_id]["status"] = "failed"
                    video_processing_jobs[job_id]["error"] = str(e)
                finally:
                    close_db_connection(connection, cursor)
                
                break
                
            elif data["status"] == "failed":
                logger.error(f"❌ AI processing failed for job {job_id}")
                video_processing_jobs[job_id]["status"] = "failed"
                video_processing_jobs[job_id]["error"] = data.get("error", "Unknown error")
                break
                
    except Exception as e:
        logger.error(f"❌ Error in poll_and_save_video: {e}")
        video_processing_jobs[job_id]["status"] = "failed"
        video_processing_jobs[job_id]["error"] = str(e)


@app.get("/api/video/process-status/{job_id}")
async def get_process_status(job_id: str):
    """Kiểm tra trạng thái xử lý video"""
    
    if job_id not in video_processing_jobs:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    
    return video_processing_jobs[job_id]


# ===== VIDEO MANAGEMENT APIs =====
@app.get("/api/videos", response_model=List[VideoResponse])
async def get_videos(
    search: Optional[str] = Query(None, description="Search by video name"),
    sort_by: Optional[str] = Query("date", description="Sort by: date, video_name, duration, status"),
    order: Optional[str] = Query("desc", description="Order: asc or desc"),
    zone_id: Optional[int] = Query(None, description="Filter by zone_id"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """Get all videos with search, filter and sort"""
    logger.info(f"📹 GET /api/videos - search={search}, sort_by={sort_by}, order={order}, zone_id={zone_id}, status={status}")
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Build query
        query = "SELECT * FROM video WHERE 1=1"
        params = []
        
        # Search
        if search:
            query += " AND video_name LIKE %s"
            params.append(f"%{search}%")
        
        # Filter by zone_id
        if zone_id is not None:
            query += " AND zone_id = %s"
            params.append(zone_id)
        
        # Filter by status
        if status:
            query += " AND status = %s"
            params.append(status)
        
        # Sort
        valid_sort_columns = ['date', 'video_name', 'duration', 'status', 'idvideo']
        if sort_by not in valid_sort_columns:
            sort_by = 'date'
        
        order = 'ASC' if order.lower() == 'asc' else 'DESC'
        query += f" ORDER BY {sort_by} {order}"
        
        cursor.execute(query, params)
        videos = cursor.fetchall()
        
        logger.info(f"✅ Retrieved {len(videos)} videos from database")
        return videos
        
    except Error as e:
        logger.error(f"❌ Error fetching videos: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        close_db_connection(connection, cursor)


@app.get("/api/videos/{video_id}", response_model=VideoResponse)
async def get_video(video_id: int):
    """Get single video by ID"""
    logger.info(f"📹 GET /api/videos/{video_id}")
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT * FROM video WHERE idvideo = %s", (video_id,))
        video = cursor.fetchone()
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return video
        
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        close_db_connection(connection, cursor)


@app.post("/api/videos", response_model=VideoResponse)
async def create_video(video: VideoCreate):
    """Create new video record"""
    logger.info(f"➕ POST /api/videos - Creating video: {video.video_name}")
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        query = """
            INSERT INTO video (video_name, zone_id, duration, date, file_path, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (
            video.video_name,
            video.zone_id,
            video.duration,
            datetime.now(),
            video.file_path,
            video.status
        )
        
        cursor.execute(query, params)
        connection.commit()
        
        video_id = cursor.lastrowid
        
        # Get created video
        cursor.execute("SELECT * FROM video WHERE idvideo = %s", (video_id,))
        created_video = cursor.fetchone()
        
        logger.info(f"✅ Video created successfully - ID: {video_id}, Name: {video.video_name}")
        return created_video
        
    except Error as e:
        logger.error(f"❌ Error creating video: {e}")
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        close_db_connection(connection, cursor)


@app.put("/api/videos/{video_id}", response_model=VideoResponse)
async def update_video(video_id: int, video: VideoCreate):
    """Update video record"""
    logger.info(f"✏️ PUT /api/videos/{video_id} - Updating video: {video.video_name}")
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Check if video exists
        cursor.execute("SELECT * FROM video WHERE idvideo = %s", (video_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Video not found")
        
        query = """
            UPDATE video 
            SET video_name = %s, zone_id = %s, duration = %s, file_path = %s, status = %s
            WHERE idvideo = %s
        """
        params = (
            video.video_name,
            video.zone_id,
            video.duration,
            video.file_path,
            video.status,
            video_id
        )
        
        cursor.execute(query, params)
        connection.commit()
        
        # Get updated video
        cursor.execute("SELECT * FROM video WHERE idvideo = %s", (video_id,))
        updated_video = cursor.fetchone()
        
        return updated_video
        
    except Error as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        close_db_connection(connection, cursor)


@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: int):
    """Delete video record"""
    logger.info(f"🗑️ DELETE /api/videos/{video_id}")
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Check if video exists
        cursor.execute("SELECT * FROM video WHERE idvideo = %s", (video_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Delete related statistics first
        cursor.execute("DELETE FROM statistic WHERE video_id = %s", (video_id,))
        
        # Delete video
        cursor.execute("DELETE FROM video WHERE idvideo = %s", (video_id,))
        connection.commit()
        
        logger.info(f"✅ Video {video_id} deleted successfully")
        return {"message": "Video deleted successfully"}
        
    except Error as e:
        logger.error(f"❌ Error deleting video {video_id}: {e}")
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        close_db_connection(connection, cursor)


@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: int):
    """Stream video file"""
    logger.info(f"🎬 Stream video request for ID: {video_id}")
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT file_path FROM video WHERE idvideo = %s", (video_id,))
        video = cursor.fetchone()
        
        if not video or not video['file_path']:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        file_path = Path(video['file_path'])
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Video file does not exist on disk")
        
        # Auto-detect MIME type based on file extension
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type or not mime_type.startswith('video/'):
            mime_type = "video/mp4"  # Fallback to mp4
        
        logger.info(f"📹 Streaming video with MIME type: {mime_type}")
        
        return FileResponse(
            path=str(file_path),
            media_type=mime_type,
            filename=file_path.name
        )
        
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        close_db_connection(connection, cursor)


@app.get("/api/videos/{video_id}/statistics")
async def get_video_statistics(video_id: int):
    """Get statistics for a specific video"""
    logger.info(f"📊 GET /api/videos/{video_id}/statistics")
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        query = """
            SELECT s.*, v.video_name, v.date as video_date
            FROM statistic s
            JOIN video v ON s.video_id = v.idvideo
            WHERE s.video_id = %s
        """
        cursor.execute(query, (video_id,))
        statistics = cursor.fetchall()
        
        return statistics
        
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        close_db_connection(connection, cursor)

@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Login endpoint - authenticate user from database"""
    logger.info(f"🔐 Login attempt for user: {username}")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Query user from account table
        query = """
            SELECT idaccount, user_name, password, role 
            FROM account 
            WHERE user_name = %s AND password = %s AND role = 'admin'
        """
        cursor.execute(query, (username, password))
        user = cursor.fetchone()
        
        if user:
            logger.info(f"✅ Login successful for user: {username} with role: {user['role']}")
            return {
                "success": True, 
                "username": user['user_name'], 
                "role": user['role'],
                "user_id": user['idaccount']
            }
        else:
            logger.warning(f"❌ Login failed for user: {username}")
            return {"success": False, "message": "Tên đăng nhập hoặc mật khẩu không đúng hoặc không có quyền admin"}
            
    except Error as e:
        logger.error(f"❌ Database error during login: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống")
    finally:
        close_db_connection(connection, cursor)