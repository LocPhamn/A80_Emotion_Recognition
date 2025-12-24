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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ai.datn.objects import EmotionClassifier

# Thêm đường dẫn ByteTrack vào sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ByteTrack'))

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


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
    """Trả về màu sắc cho từng cảm xúc"""
    emotion_colors = {
        'angry': (0, 0, 255),  # Đỏ
        'disgust': (0, 128, 0),  # Xanh lá đậm
        'fear': (128, 0, 128),  # Tím
        'happy': (0, 255, 255),  # Vàng
        'neutral': (200, 200, 200),  # Xám
        'sad': (255, 0, 0),  # Xanh dương
        'surprise': (255, 165, 0)  # Cam
    }
    return emotion_colors.get(emotion, (255, 255, 255))


def visualize_tracking_with_emotion(frame, online_targets, emotion_classifier, track_emotions, fps=0, predict_emotion=True):
    """Vẽ kết quả tracking và cảm xúc lên frame"""
    # Vẽ FPS và số lượng tracks
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {len(online_targets)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

        face_crop = frame[y1:y2, x1:x2]

        # ✅ CHỈ PREDICT KHI FLAG = TRUE
        if predict_emotion and face_crop.size > 0 and w > 20 and h > 20:
            emotion, emotion_conf = emotion_classifier.predict(face_crop)
            track_emotions[track_id] = (emotion, emotion_conf)
        else:
            # Dùng emotion cached
            if track_id in track_emotions:
                emotion, emotion_conf = track_emotions[track_id]
            else:
                emotion, emotion_conf = "unknown", 0.0

        color = get_emotion_color(emotion)

        # Vẽ bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Vẽ labels
        id_label = f"ID:{track_id}"
        emotion_label = f"{emotion}"
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
        
        # ✅ TỐI ƯU: Set GPU nếu có
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("✅ Using GPU for YOLO")
        
        # ✅ TỐI ƯU: Giảm kích thước input YOLO
        self.input_size = 416  # Thay vì 640 default
        
        print(f"Loading Emotion Classifier: {emotion_weights_path}")
        self.emotion_classifier = EmotionClassifier(emotion_weights_path)
        
        # ✅ TỐI ƯU: Cache emotion để không predict mỗi frame
        self.emotion_cache_frames = 3  # Chỉ predict emotion mỗi 3 frames
        self.frame_count = 0

        # Khởi tạo ByteTracker
        args = Args()
        args.track_thresh = track_thresh
        args.track_buffer = track_buffer
        args.match_thresh = match_thresh

        self.tracker = BYTETracker(args, frame_rate=30)

        self.track_emotions = {}
        self.track_history = defaultdict(lambda: [])
        self.show_trajectory = show_trajectory

        self.prev_time = time.time()
        self.fps = 0
        
        print("✅ FaceEmotionTracker initialized!")

    def process_frame(self, frame):
        """Xử lý một frame và trả về kết quả"""
        # Tính FPS
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time + 1e-6)
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

        # ✅ CACHE EMOTION - chỉ predict mỗi N frames
        should_predict_emotion = (self.frame_count % self.emotion_cache_frames == 0)
        
        # Visualize với emotion
        visualize_frame = visualize_tracking_with_emotion(
            frame, online_targets, self.emotion_classifier,
            self.track_emotions, int(self.fps),
            predict_emotion=should_predict_emotion
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

    def reset(self):
        """Reset tracker state"""
        self.tracker = BYTETracker(Args(), frame_rate=30)
        self.track_emotions = {}
        self.track_history = defaultdict(lambda: [])
        print("Tracker reset!")


if __name__ == "__main__":
    pass