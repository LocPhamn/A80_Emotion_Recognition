from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import websockets
import asyncio
import json
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from typing import Optional, List
import logging
import requests
import os
from pathlib import Path
import uuid

# Import models
from models import VideoResponse, VideoCreate, VideoUpdate, StatisticResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ video format: mp4, avi, mov, mkv")
    
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
                
                logger.info(f"✅ Video saved: {output_path}")
                
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
                        str(output_path),
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
