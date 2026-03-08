import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import defaultdict
import argparse
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import unicodedata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ai.datn.objects import EmotionClassifier

# Thêm đường dẫn ByteTrack vào sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ByteTrack'))

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


def remove_vietnamese_accents(text):
    """
    Convert tiếng Việt có dấu → không dấu để OpenCV hiển thị được
    
    Args:
        text: String tiếng Việt có dấu
    
    Returns:
        String không dấu
    """
    # Mapping đặc biệt cho các ký tự tiếng Việt
    vietnamese_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'đ': 'd',
        'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
        'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
        'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
        'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
        'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
        'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
        'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
        'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
        'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
        'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
        'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
        'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
        'Đ': 'D'
    }
    
    result = ''
    for char in text:
        result += vietnamese_map.get(char, char)
    
    return result

# ====== HỖ TRỢ CHO BYTETRACKER ======
class Args:
    """Class để lưu trữ các tham số cho ByteTracker"""

    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.mot20 = False
        self.min_box_area = 10


def yolo_detections_to_bytetrack_format(results, img_shape):
    """Chuyển đổi kết quả detection từ YOLOv11 sang format của ByteTrack"""
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return torch.empty((0, 5))

    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu()
    confidences = boxes.conf.cpu()

    output_results = torch.cat([
        xyxy,
        confidences.reshape(-1, 1)
    ], dim=1)

    return output_results


def get_color(idx):
    """Tạo màu sắc duy nhất cho mỗi ID"""
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def get_emotion_color(emotion):
    """Trả về màu sắc cho từng cảm xúc (hỗ trợ cả tiếng Việt và tiếng Anh)"""
    # Mapping tiếng Việt sang tiếng Anh
    vi_to_en = {
        'tức giận': 'angry',
        'khó chịu': 'disgust',
        'sợ hãi': 'fear',
        'hạnh phúc': 'happy',
        'trung tính': 'neutral',
        'buồn bã': 'sad',
        'bất ngờ': 'surprise'
    }
    
    # Convert sang tiếng Anh nếu là tiếng Việt
    emotion_en = vi_to_en.get(emotion, emotion)
    
    # Màu sắc BGR cho OpenCV (rực rỡ hơn để dễ nhìn)
    emotion_colors = {
        'angry': (0, 0, 255),        # Đỏ tươi
        'disgust': (255, 0, 255),    # Magenta/Tím hồng
        'fear': (128, 0, 128),       # Tím đậm
        'happy': (0, 255, 0),        # Xanh lá tươi
        'neutral': (180, 180, 180),  # Xám nhạt
        'sad': (255, 100, 0),        # Xanh dương
        'surprise': (0, 255, 255)    # Vàng tươi (cyan)
    }
    return emotion_colors.get(emotion_en, (255, 255, 255))


def visualize_tracking_with_emotion(frame, online_targets, emotion_classifier, track_emotions, fps=0, predict_emotion=True, get_stable_emotion_func=None):
    """Vẽ kết quả tracking và cảm xúc lên frame"""
    # Vẽ FPS và số lượng tracks
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {len(online_targets)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # BATCH PROCESSING - Thu thập tất cả face crops trước
    if predict_emotion and len(online_targets) > 0:
        face_crops = []
        valid_tracks = []
        
        for track in online_targets:
            tlwh = track.tlwh
            x1, y1, w, h = tlwh
            x2 = x1 + w
            y2 = y1 + h
            
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0 and w > 20 and h > 20:
                face_crops.append(face_crop)
                valid_tracks.append(track.track_id)
        
        # Predict tất cả faces cùng lúc (BATCH)
        if len(face_crops) > 0:
            emotion_results = emotion_classifier.predict_batch(face_crops)
            
            # Lưu kết quả vào cache với smoothing
            for track_id, (emotion, conf) in zip(valid_tracks, emotion_results):
                # Sử dụng stable emotion nếu có hàm smoothing
                if get_stable_emotion_func:
                    stable_emotion, stable_conf = get_stable_emotion_func(track_id, emotion, conf)
                    track_emotions[track_id] = (stable_emotion, stable_conf)
                else:
                    track_emotions[track_id] = (emotion, conf)

    # Vẽ visualization cho từng track
    for track in online_targets:
        tlwh = track.tlwh
        track_id = track.track_id
        score = track.score

        x1, y1, w, h = tlwh
        x2 = x1 + w
        y2 = y1 + h

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2))

        # Lấy emotion từ cache
        if track_id in track_emotions:
            emotion, emotion_conf = track_emotions[track_id]
        else:
            emotion, emotion_conf = "unknown", 0.0

        color = get_emotion_color(emotion)

        # Vẽ bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Vẽ labels
        id_label = f"ID:{track_id}"
        emotion_label = remove_vietnamese_accents(emotion)
        conf_label = f"{emotion_conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (id_w, id_h), _ = cv2.getTextSize(id_label, font, 0.6, 2)
        (emo_w, emo_h), _ = cv2.getTextSize(emotion_label, font, 0.7, 2)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_label, font, 0.5, 1)

        cv2.rectangle(frame, (x1, y1 - id_h - emo_h - 25),
                      (x1 + max(id_w, emo_w) + 15, y1), color, -1)

        cv2.putText(frame, id_label, (x1 + 5, y1 - emo_h - 15),
                    font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, emotion_label, (x1 + 5, y1 - 5),
                    font, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (x2 - conf_w - 10, y2),
                      (x2, y2 + conf_h + 10), color, -1)
        cv2.putText(frame, conf_label, (x2 - conf_w - 5, y2 + conf_h + 5),
                    font, 0.5, (255, 255, 255), 1)

        # Vẽ center point
        center_x = int(x1 + (x2 - x1) / 2)
        center_y = int(y1 + (y2 - y1) / 2)
        cv2.circle(frame, (center_x, center_y), 5, color, -1)

    return frame

# ====== MAIN TRACKER CLASS ======
class FaceEmotionTracker:
    """Class để tracking và nhận diện cảm xúc từ frame"""

    def __init__(
            self,
            model_path=r"D:\Python plus\AI_For_CV\script\datn-backed\ai\datn\model_weights\yolo_models\yolov11s_custom.pt",
            emotion_weights_path=r"D:\Python plus\AI_For_CV\script\datn-backed\ai\datn\model_weights\mobilenet_models\mobilenetv3_best_weights_only.pth",
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            show_trajectory=False
    ):
        """Khởi tạo tracker"""
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        if torch.cuda.is_available():
            self.model.to('cuda')
            print("đang sử dụng GPU cho YOLO")

        self.input_size = 640

        print(f"Khởi động Emotion Classifier: {emotion_weights_path}")
        # Emotion labels tiếng Việt: ['tức giận', 'khó chịu', 'sợ hãi', 'hạnh phúc', 'trung tính', 'buồn bã', 'bất ngờ']

        self.penalties = [1.0, 1.0, 1.0, 1.0, 0.01, 1.0, 2.0]

        self.emotion_classifier = EmotionClassifier(emotion_weights_path, emotion_penalties=self.penalties)
        
        # Mapping tiếng Việt sang tiếng Anh (cho database)
        self.emotion_to_english = {
            'tức giận': 'angry',
            'khó chịu': 'disgust',
            'sợ hãi': 'fear',
            'hạnh phúc': 'happy',
            'trung tính': 'neutral',
            'buồn bã': 'sad',
            'bất ngờ': 'surprise'
        }
        
        self.emotion_cache_frames = 10  # Cập nhật emotion mỗi 10 frames
        self.frame_count = 0

        # Khởi tạo ByteTracker
        args = Args()
        args.track_thresh = track_thresh
        args.track_buffer = track_buffer
        args.match_thresh = match_thresh

        self.tracker = BYTETracker(args, frame_rate=30)

        self.track_emotions = {}  # Current emotion display
        self.emotion_history = defaultdict(lambda: [])  # Lịch sử emotions cho smoothing
        self.emotion_history_size = 7  # Lưu 10 predictions gần nhất
        self.track_history = defaultdict(lambda: [])
        self.show_trajectory = show_trajectory

        self.prev_time = time.time()
        self.fps = 0
        self.fps_history = []  # Lưu lịch sử FPS
        self.fps_avg_window = 30  # Trung bình 30 frames

        print("Đã khởi tạo FaceEmotionTracker!")

    def get_stable_emotion(self, track_id, new_emotion, new_confidence):
        """Lấy emotion ổn định bằng majority voting từ lịch sử"""
        # Thêm emotion mới vào history
        self.emotion_history[track_id].append((new_emotion, new_confidence))
        
        # Giới hạn size của history
        if len(self.emotion_history[track_id]) > self.emotion_history_size:
            self.emotion_history[track_id].pop(0)
        
        # Nếu history còn ít, trả về emotion mới
        if len(self.emotion_history[track_id]) < 3:
            return new_emotion, new_confidence
        
        # Đếm số lần xuất hiện của mỗi emotion trong history
        from collections import Counter
        emotion_counts = Counter([emo for emo, conf in self.emotion_history[track_id]])
        
        # Lấy emotion xuất hiện nhiều nhất (majority voting)
        most_common_emotion = emotion_counts.most_common(1)[0][0]
        
        # Tính confidence trung bình của emotion được chọn
        confidences = [conf for emo, conf in self.emotion_history[track_id] if emo == most_common_emotion]
        avg_confidence = sum(confidences) / len(confidences) if confidences else new_confidence
        
        return most_common_emotion, avg_confidence

    def process_frame(self, frame):
        """Xử lý một frame và trả về kết quả"""
        curr_time = time.time()
        delta_time = curr_time - self.prev_time

        if delta_time > 0:
            instant_fps = 1.0 / delta_time
            self.fps_history.append(instant_fps)

            if len(self.fps_history) > self.fps_avg_window:
                self.fps_history.pop(0)

            self.fps = sum(self.fps_history) / len(self.fps_history)

        self.prev_time = curr_time
        self.frame_count += 1

        height, width = frame.shape[:2]

        # YOLO Detection với size nhỏ hơn
        results = self.model(frame, imgsz=self.input_size, verbose=False)

        # Chuyển đổi sang format ByteTrack
        detections = yolo_detections_to_bytetrack_format(results, frame.shape[:2])

        # ByteTrack Tracking
        img_info = (height, width)
        img_size = (height, width)

        if len(detections) > 0:
            online_targets = self.tracker.update(detections, img_info, img_size)
        else:
            online_targets = []

        # CACHE EMOTION - chỉ predict mỗi N frames
        should_predict_emotion = (self.frame_count % self.emotion_cache_frames == 0)

        # hiển thị với emotion
        visualize_frame = visualize_tracking_with_emotion(
            frame, online_targets, self.emotion_classifier,
            self.track_emotions, int(self.fps),
            predict_emotion=should_predict_emotion,
            get_stable_emotion_func=self.get_stable_emotion
        )

        # Tạo metadata
        tracks_info = []
        for track in online_targets:
            track_id = track.track_id
            tlwh = track.tlwh
            emotion, emotion_conf = self.track_emotions.get(track_id, ("unknown", 0.0))

            tracks_info.append({
                'id': int(track_id),
                'bbox': {
                    'x': float(tlwh[0]),
                    'y': float(tlwh[1]),
                    'width': float(tlwh[2]),
                    'height': float(tlwh[3])
                },
                'emotion': emotion,
                'confidence': float(emotion_conf),
                'color': get_emotion_color(emotion)
            })

        return {
            'frame': visualize_frame,
            'fps': float(self.fps),
            'tracks': tracks_info
        }

# ====== RESET FUNCTION ======
    def reset(self):
        """Reset tracker state"""
        self.tracker = BYTETracker(Args(), frame_rate=30)
        self.track_emotions = {}
        self.emotion_history = defaultdict(lambda: [])
        self.track_history = defaultdict(lambda: [])
        print("Tracker reset!")


# ====== VIDEO PROCESSING FUNCTION ======
    def process_video(
        input_video_path,
        output_video_path=None,
        yolo_model_path=r"D:\Python plus\AI_For_CV\script\datn-backed\ai\datn\model_weights\yolo_models\yolov11s_custom.pt",
        emotion_model_path=r"D:\Python plus\AI_For_CV\script\datn-backed\ai\datn\model_weights\mobilenet_models\mobilenetv3_best_weights_only.pth",
        show_preview=False,
        skip_frames=1
    ):
        """
        Xử lý video với face tracking và emotion detection

        Args:
            input_video_path: Đường dẫn đến video input
            output_video_path: Đường dẫn lưu video output (mặc định: input_processed.mp4)
            yolo_model_path: Đường dẫn model YOLO (optional)
            emotion_model_path: Đường dẫn model emotion (optional)
            show_preview: Hiển thị preview trong khi xử lý
            skip_frames: Bỏ qua N frames để tăng tốc (1 = xử lý tất cả)

        Returns:
            Dict chứa thông tin xử lý
        """
        # Track unique visitor IDs và emotion của họ
        visited_ids = set()  # Lưu các track ID duy nhất
        track_final_emotion = {}  # Lưu emotion CUỐI CÙNG cho mỗi ID (đơn giản nhất)
        
        # Kiểm tra file input
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video không tồn tại: {input_video_path}")

        # Tạo output path nếu chưa có
        if output_video_path is None:
            input_path = Path(input_video_path)
            output_video_path = str(input_path.parent / f"{input_path.stem}_processed{input_path.suffix}")

        # Tạo thư mục output nếu chưa có
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

        print(f"Input video: {input_video_path}")
        print(f"Output video: {output_video_path}")

        # Mở video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {input_video_path}")

        # Lấy thông tin video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Thời lượng: {total_frames/fps:.2f} giây")

        # Khởi tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # hoặc 'XVID', 'H264'
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise ValueError(f"Không thể tạo video output: {output_video_path}")

        # Khởi tạo tracker
        print(f"Đang khởi tạo FaceEmotionTracker...")
        tracker_kwargs = {}
        if yolo_model_path:
            tracker_kwargs['model_path'] = yolo_model_path
        if emotion_model_path:
            tracker_kwargs['emotion_weights_path'] = emotion_model_path
        
        tracker = FaceEmotionTracker(**tracker_kwargs)
        
        # Xử lý từng frame
        print(f"Bắt đầu xử lý video...")
        frame_idx = 0
        processed_frames = 0
        
        try:
            try:
                from tqdm import tqdm
                use_tqdm = True
                pbar = tqdm(total=total_frames, desc="Processing", unit="frames")
            except ImportError:
                use_tqdm = False
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_idx += 1
                
                # Skip frames nếu cần
                if skip_frames > 1 and frame_idx % skip_frames != 0:
                    out.write(frame)
                    if use_tqdm:
                        pbar.update(1)
                    elif frame_idx % 100 == 0:
                        print(f"  Processed: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
                    continue
                
                # Xử lý frame
                result = tracker.process_frame(frame)
                processed_frame = result['frame']
                
                # Đếm unique visitors và lưu emotion cuối cùng của họ
                for track in result['tracks']:
                    track_id = track['id']
                    emotion_vi = track['emotion']  # Emotion tiếng Việt từ model
                    
                    # Thêm ID vào set unique visitors
                    visited_ids.add(track_id)
                    
                    # Convert emotion sang tiếng Anh để lưu vào database
                    emotion_en = tracker.emotion_to_english.get(emotion_vi, 'neutral')
                    
                    # Debug: In ra để kiểm tra (chỉ frame đầu tiên)
                    if frame_idx == 1 and track_id not in track_final_emotion:
                        print(f"[DEBUG] Track {track_id}: {emotion_vi} → {emotion_en}")
                    
                    # Lưu emotion CUỐI CÙNG cho ID này (ghi đè)
                    track_final_emotion[track_id] = emotion_en
                
                # Ghi frame đã xử lý
                out.write(processed_frame)
                processed_frames += 1
                
                # Hiển thị preview nếu cần
                if show_preview:
                    cv2.imshow('Processing Video', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nNgười dùng dừng xử lý!")
                        break
                
                # Cập nhật progress
                if use_tqdm:
                    pbar.update(1)
                    pbar.set_postfix({
                        'FPS': f"{result['fps']:.1f}",
                        'Faces': len(result['tracks'])
                    })
                elif frame_idx % 100 == 0:
                    print(f"  Processed: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%) - FPS: {result['fps']:.1f}, Faces: {len(result['tracks'])}")
        
        except KeyboardInterrupt:
            print("\n Đã dừng bởi người dùng (Ctrl+C)")
        
        finally:
            # Cleanup
            if use_tqdm:
                pbar.close()
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Tính tổng khách duy nhất
        total_visitor = len(visited_ids)
        
        # Đếm số người cho mỗi emotion - LOGIC ĐƠN GIẢN
        emotion_visitors = defaultdict(int)
        
        print(f"\n🔍 DEBUG - Processing {total_visitor} visitors...")
        
        for track_id in visited_ids:
            if track_id in track_final_emotion:
                emotion = track_final_emotion[track_id]
                emotion_visitors[emotion] += 1  # Mỗi người đếm 1 lần duy nhất
                
                # Debug: in ra 5 người đầu tiên
                if len(emotion_visitors) <= 5:
                    print(f"   Track {track_id}: {emotion}")
            else:
                # Nếu không có emotion (không nên xảy ra), gán neutral
                print(f"⚠️ WARNING: Track {track_id} không có emotion, gán 'neutral'")
                emotion_visitors['neutral'] += 1
        
        # Validation: kiểm tra tổng phải bằng số người
        total_counted = sum(emotion_visitors.values())
        if total_counted != total_visitor:
            print(f"⚠️ WARNING: Tổng emotion count ({total_counted}) ≠ total visitors ({total_visitor})!")
        
        # Tính tỉ lệ cảm xúc (tính theo phần trăm %)
        emotion_ratios = {}
        for emotion_name in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
            count = emotion_visitors.get(emotion_name, 0)
            ratio = (count / total_visitor * 100) if total_visitor > 0 else 0.0
            emotion_ratios[emotion_name] = {
                'ratio': round(ratio, 2),
                'count': count
            }
        
        
        # Tổng kết
        print(f"\n✅ Hoàn thành!")
        print(f"Đã xử lý: {processed_frames}/{total_frames} frames")
        print(f"Video đã lưu tại: {output_video_path}")
        print(f"\n📊 THỐNG KÊ:")
        print(f"   Tổng số người: {total_visitor}")
        
        # Debug: hiển thị raw data
        print(f"\n🔍 DEBUG - Emotion count:")
        for emotion_en, count in sorted(emotion_visitors.items()):
            print(f"   {emotion_en}: {count} người")
        
        print(f"\n📊 Tỉ lệ cảm xúc (%):")
        
        # Mapping hiển thị tiếng Việt cho người dùng
        emotion_en_to_vi = {
            'angry': 'Tức giận',
            'disgust': 'Khó chịu',
            'fear': 'Sợ hãi',
            'happy': 'Hạnh phúc',
            'neutral': 'Trung tính',
            'sad': 'Buồn bã',
            'surprise': 'Bất ngờ'
        }
        
        total_percentage = 0.0
        for emotion_en, data in sorted(emotion_ratios.items()):
            emotion_vi = emotion_en_to_vi.get(emotion_en, emotion_en)
            print(f"   - {emotion_vi:15s}: {data['ratio']:6.2f}% ({data['count']} người)")
            total_percentage += data['ratio']
        
        print(f"   ✓ Tổng: {round(total_percentage, 2)}% (phải = 100%)")
        
        # Validation: Kiểm tra tổng phần trăm phải gần bằng 100%
        if abs(total_percentage - 100.0) > 0.1:
            print(f"⚠️ WARNING: Tổng phần trăm không đúng ({total_percentage}%)! Có thể có lỗi logic.")
        
        return {
            'total_visitor': total_visitor,
            'emotion_ratios': emotion_ratios,
            'input_path': input_video_path,
            'output_path': output_video_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'resolution': (width, height)
        }


if __name__ == "__main__":
   pass