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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ai.datn.objects import EmotionClassifier

# Thêm đường dẫn ByteTrack vào sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ByteTrack'))

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

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
            
            # Lưu kết quả vào cache
            for track_id, (emotion, conf) in zip(valid_tracks, emotion_results):
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
            print("✅ Using GPU for YOLO")
        
        self.input_size = 960
        
        print(f"Loading Emotion Classifier: {emotion_weights_path}")
        self.emotion_classifier = EmotionClassifier(emotion_weights_path)
        
 
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
        self.fps_history = []  # Lưu lịch sử FPS
        self.fps_avg_window = 30  # Trung bình 30 frames
        
        print("Đã khởi tạo FaceEmotionTracker!")

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
        
# ====== RESET FUNCTION ======
    def reset(self):
        """Reset tracker state"""
        self.tracker = BYTETracker(Args(), frame_rate=30)
        self.track_emotions = {}
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
        
        # Tổng kết
        print(f"\nHoàn thành!")
        print(f"Đã xử lý: {processed_frames}/{total_frames} frames")
        print(f"Video đã lưu tại: {output_video_path}")
        
        return {
            'input_path': input_video_path,
            'output_path': output_video_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'resolution': (width, height)
        }


if __name__ == "__main__":
   pass