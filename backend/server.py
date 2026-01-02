from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import websockets
import asyncio
import json
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
import logging

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

# Pydantic models
class VideoResponse(BaseModel):
    idvideo: int
    video_name: str
    zone_id: Optional[int]
    duration: Optional[int]
    date: Optional[datetime]
    file_path: Optional[str]
    status: Optional[str]

class VideoCreate(BaseModel):
    video_name: str
    zone_id: Optional[int] = None
    duration: Optional[int] = None
    file_path: Optional[str] = None
    status: Optional[str] = "pending"

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
