"""
Configuration file for Webcam Demo Local
Chỉnh settings ở đây thay vì sửa code chính
"""

# ===== WEBCAM SETTINGS =====
WEBCAM_CONFIG = {
    # ID của webcam (0 = webcam mặc định)
    # Thử 1, 2, 3... nếu webcam 0 không hoạt động
    "webcam_id": 0,
    
    # Resolution hiển thị
    "display_width": 1280,
    "display_height": 720,
    
    # FPS recording (15-30 tùy performance)
    "recording_fps": 15.0,
}

# ===== MODEL SETTINGS =====
MODEL_CONFIG = {
    # Đường dẫn model YOLO
    "yolo_model_path": r"D:\Python plus\AI_For_CV\script\datn-backed\ai\datn\model_weights\yolo_models\yolov11s_custom.pt",
    
    # Đường dẫn model Emotion
    "emotion_model_path": r"D:\Python plus\AI_For_CV\script\datn-backed\ai\datn\model_weights\mobilenet_models\mobilenetv3_best_weights_only.pth",
    
    # YOLO input size (để tăng tốc độ)
    "yolo_input_size": 640,
    
    # Emotion cache frames (predict emotion mỗi N frames)
    # Tăng lên để tăng tốc độ (trade-off: cập nhật chậm hơn)
    "emotion_cache_frames": 10,
}

# ===== TRACKING SETTINGS =====
TRACKING_CONFIG = {
    # ByteTracker threshold
    "track_thresh": 0.5,
    "track_buffer": 30,
    "match_thresh": 0.8,
    
    # Emotion smoothing (majority voting window size)
    "emotion_history_size": 7,
}

# ===== OUTPUT SETTINGS =====
OUTPUT_CONFIG = {
    # Thư mục lưu recordings và screenshots
    "output_dir": "./demo_recordings",
    
    # Format tên file
    "video_filename_format": "demo_{timestamp}.mp4",
    "screenshot_filename_format": "screenshot_{timestamp}.jpg",
    
    # Video codec ('mp4v', 'XVID', 'H264')
    "video_codec": "mp4v",
}

# ===== DISPLAY SETTINGS =====
DISPLAY_CONFIG = {
    # Hiển thị trajectory (đường đi của face)
    "show_trajectory": False,
    
    # Hiển thị grid/debug info
    "show_debug_info": False,
    
    # Font scale cho text trên màn hình
    "font_scale": 0.6,
    
    # Recording indicator settings
    "recording_indicator": {
        "position": (30, 30),
        "radius": 15,
        "color": (0, 0, 255),  # BGR: Red
        "blink_frequency": 2,  # Nhấp nháy 2 lần/giây
    }
}

# ===== PERFORMANCE SETTINGS =====
PERFORMANCE_CONFIG = {
    # Skip frames nếu xử lý quá chậm (tương tự AI server)
    "frame_skip_threshold": 3,
    
    # GPU memory management
    "clear_gpu_cache": True,
    "gpu_cache_clear_interval": 100,  # Clear mỗi 100 frames
}

# ===== EMOTION LABELS =====
EMOTION_LABELS = {
    "vi_to_en": {
        'tức giận': 'angry',
        'khó chịu': 'disgust',
        'sợ hãi': 'fear',
        'hạnh phúc': 'happy',
        'trung tính': 'neutral',
        'buồn bã': 'sad',
        'bất ngờ': 'surprise'
    },
    "display_labels": {
        'angry': 'Tức giận',
        'disgust': 'Chán nản',
        'fear': 'Sợ hãi',
        'happy': 'Hạnh phúc',
        'neutral': 'Trung tính',
        'sad': 'Buồn bã',
        'surprise': 'Bất ngờ'
    }
}

# ===== EMOTION PENALTIES =====
# Adjust confidence scores để giảm bias
# Format: [angry, disgust, fear, happy, neutral, sad, surprise]
EMOTION_PENALTIES = [1.0, 2.0, 2.0, 1.0, 0.01, 1.0, 2.0]
# neutral = 0.01 để giảm mạnh cảm xúc trung tính
