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


def visualize_tracking_with_emotion(frame, online_targets, emotion_classifier, track_emotions, fps=0):
    """
    Vẽ kết quả tracking và cảm xúc lên frame

    Args:
        frame: Frame ảnh
        online_targets: List các track từ ByteTracker
        emotion_classifier: EmotionClassifier instance
        track_emotions: Dict lưu cảm xúc của mỗi track_id
        fps: FPS hiện tại

    Returns:
        frame: Frame đã được vẽ
    """
    # Vẽ FPS và số lượng tracks
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {len(online_targets)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for track in online_targets:
        tlwh = track.tlwh
        track_id = track.track_id
        score = track.score

        # Chuyển từ tlwh sang xyxy
        x1, y1, w, h = tlwh
        x2 = x1 + w
        y2 = y1 + h

        # Đảm bảo tọa độ nằm trong frame
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2))

        # Crop khuôn mặt
        face_crop = frame[y1:y2, x1:x2]

        # Dự đoán cảm xúc nếu crop hợp lệ
        if face_crop.size > 0 and w > 20 and h > 20:
            emotion, emotion_conf = emotion_classifier.predict(face_crop)
            # Lưu cảm xúc cho track này
            track_emotions[track_id] = (emotion, emotion_conf)
        else:
            # Sử dụng cảm xúc trước đó nếu có
            if track_id in track_emotions:
                emotion, emotion_conf = track_emotions[track_id]
            else:
                emotion, emotion_conf = "unknown", 0.0

        # Lấy màu theo cảm xúc
        color = get_emotion_color(emotion)

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Tạo label với ID và cảm xúc
        id_label = f"ID:{track_id}"
        emotion_label = f"{emotion}"
        conf_label = f"{emotion_conf:.2f}"

        # Tính kích thước text
        font = cv2.FONT_HERSHEY_SIMPLEX
        (id_w, id_h), _ = cv2.getTextSize(id_label, font, 0.6, 2)
        (emo_w, emo_h), _ = cv2.getTextSize(emotion_label, font, 0.7, 2)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_label, font, 0.5, 1)

        # Vẽ background cho ID (trên cùng)
        cv2.rectangle(frame,
                      (x1, y1 - id_h - emo_h - 25),
                      (x1 + max(id_w, emo_w) + 15, y1),
                      color, -1)

        # Vẽ ID
        cv2.putText(frame, id_label, (x1 + 5, y1 - emo_h - 15),
                    font, 0.6, (255, 255, 255), 2)

        # Vẽ cảm xúc (dưới ID)
        cv2.putText(frame, emotion_label, (x1 + 5, y1 - 5),
                    font, 0.7, (255, 255, 255), 2)

        # Vẽ confidence ở góc dưới phải
        cv2.rectangle(frame,
                      (x2 - conf_w - 10, y2),
                      (x2, y2 + conf_h + 10),
                      color, -1)
        cv2.putText(frame, conf_label, (x2 - conf_w - 5, y2 + conf_h + 5),
                    font, 0.5, (255, 255, 255), 1)

        # Vẽ điểm center
        center_x = int(x1 + (x2 - x1) / 2)
        center_y = int(y1 + (y2 - y1) / 2)
        cv2.circle(frame, (center_x, center_y), 5, color, -1)

    return frame


def main():
    parser = argparse.ArgumentParser("YOLOv11 + ByteTrack Demo")
    parser.add_argument("--video", "-v", type=str, default="train_videos/IMG_5279.MOV",
                        help="Path to video file")
    parser.add_argument("--model", "-m", type=str, default="yolo_models/yolov11s_custom.pt",
                        help="Path to YOLO model")
    parser.add_argument("--emotion_weights", "-e", type=str, required=True,
                        help="Path to emotion model weights (MobileNetV3-Large)")
    parser.add_argument("--track_thresh", "-t", type=float, default=0.5,
                        help="Tracking confidence threshold")
    parser.add_argument("--track_buffer", "-b", type=int, default=30,
                        help="Frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8,
                        help="Matching threshold for tracking")
    parser.add_argument("--save", action="store_true",
                        help="Save output video")
    parser.add_argument("--output", type=str, default="output_emotion.mp4",
                        help="Output video path")
    parser.add_argument("--show_trajectory", action="store_true",
                        help="Show tracking trajectory")

    args_parser = parser.parse_args()

    # Khởi tạo YOLO model
    print(f"Loading YOLO model: {args_parser.model}")
    model = YOLO(args_parser.model)

    # Khởi tạo Emotion Classifier
    emotion_classifier = EmotionClassifier(args_parser.emotion_weights)

    # Khởi tạo ByteTracker
    args = Args()
    args.track_thresh = args_parser.track_thresh
    args.track_buffer = args_parser.track_buffer
    args.match_thresh = args_parser.match_thresh

    tracker = BYTETracker(args, frame_rate=30)

    # Dictionary để lưu cảm xúc của mỗi track
    track_emotions = {}
    track_history = defaultdict(lambda: [])

    # Mở video
    cap = cv2.VideoCapture(args_parser.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args_parser.video}")
        return

    # Lấy thông tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video info: {width}x{height} @ {fps_video} FPS")

    # Khởi tạo video writer nếu cần save
    if args_parser.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args_parser.output, fourcc, fps_video, (width, height))
        print(f"Saving output to: {args_parser.output}")

    fps = 0
    prev_time = time.time()
    frame_id = 0

    print("\nProcessing video... Press 'q' to quit")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Tính FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # YOLO Detection
        results = model(frame, verbose=False)

        # Chuyển đổi sang format ByteTrack
        detections = yolo_detections_to_bytetrack_format(results, frame.shape[:2])

        # ByteTrack Tracking
        img_info = (height, width)
        img_size = (height, width)

        if len(detections) > 0:
            online_targets = tracker.update(detections, img_info, img_size)
        else:
            online_targets = []

        # Trajectory
        if args_parser.show_trajectory:
            for track in online_targets:
                track_id = track.track_id
                tlwh = track.tlwh
                center_x = tlwh[0] + tlwh[2] / 2
                center_y = tlwh[1] + tlwh[3] / 2
                track_history[track_id].append((int(center_x), int(center_y)))

                if len(track_history[track_id]) > 50:
                    track_history[track_id].pop(0)

            for track_id, points in track_history.items():
                if len(points) > 1:
                    emotion = track_emotions.get(track_id, ("unknown", 0.0))[0]
                    color = get_emotion_color(emotion)
                    points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points_array], isClosed=False,
                                  color=color, thickness=2, lineType=cv2.LINE_AA)

        # Visualize với emotion
        frame = visualize_tracking_with_emotion(frame, online_targets, emotion_classifier,
                                                track_emotions, fps)

        # Hiển thị thông tin
        if frame_id % 30 == 0:
            num_dets = len(detections) if isinstance(detections, torch.Tensor) else detections.shape[0]
            print(f"Frame {frame_id}: Detections={num_dets}, "
                  f"Tracks={len(online_targets)}, FPS={fps:.1f}")

        # Hiển thị frame
        cv2.imshow("Face Tracking + Emotion Recognition", frame)

        # Lưu video nếu cần
        if args_parser.save:
            out.write(frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    if args_parser.save:
        out.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_id} frames")
    print("Done!")


def run_webcam():
    """Chạy tracking với webcam và emotion recognition"""
    parser = argparse.ArgumentParser("YOLOv11 + ByteTrack + Emotion Webcam")
    parser.add_argument("--model", "-m", type=str,
                        default=r"D:\Python plus\AI_For_CV\script\datn-backed\ai\datn\model_weights\yolo_models\yolov11s_custom.pt",
                        help="Path to YOLO model")
    parser.add_argument("--emotion_weights", "-e", type=str, required=True,
                        help="Path to emotion model weights (MobileNetV3-Large)")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--track_thresh", "-t", type=float, default=0.5,
                        help="Tracking confidence threshold")
    parser.add_argument("--track_buffer", "-b", type=int, default=30,
                        help="Frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8,
                        help="Matching threshold for tracking")
    parser.add_argument("--save", action="store_true",
                        help="Save output video")
    parser.add_argument("--output", type=str, default="webcam_emotion_output.mp4",
                        help="Output video path")
    parser.add_argument("--show_trajectory", action="store_true",
                        help="Show tracking trajectory")

    args_parser = parser.parse_args()

    # Khởi tạo YOLO model
    print(f"Loading YOLO model: {args_parser.model}")
    model = YOLO(args_parser.model)

    # Khởi tạo Emotion Classifier
    emotion_classifier = EmotionClassifier(args_parser.emotion_weights)

    # Khởi tạo ByteTracker
    args = Args()
    args.track_thresh = args_parser.track_thresh
    args.track_buffer = args_parser.track_buffer
    args.match_thresh = args_parser.match_thresh

    tracker = BYTETracker(args, frame_rate=30)

    # Dictionary để lưu cảm xúc và trajectory
    track_emotions = {}
    track_history = defaultdict(lambda: [])

    # Mở webcam
    print(f"Opening camera {args_parser.camera}...")
    cap = cv2.VideoCapture(args_parser.camera)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {args_parser.camera}")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Lấy thông tin camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera info: {width}x{height}")

    # Khởi tạo video writer nếu cần
    if args_parser.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args_parser.output, fourcc, 20, (width, height))
        print(f"Saving output to: {args_parser.output}")

    fps = 0
    prev_time = time.time()
    frame_id = 0

    print("\nWebcam running... Press 'q' to quit, 's' to save screenshot")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_id += 1

        # Tính FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # YOLO Detection
        results = model(frame, verbose=False)

        # Chuyển đổi sang format ByteTrack
        detections = yolo_detections_to_bytetrack_format(results, frame.shape[:2])

        # ByteTrack Tracking
        img_info = (height, width)
        img_size = (height, width)

        if len(detections) > 0:
            online_targets = tracker.update(detections, img_info, img_size)
        else:
            online_targets = []

        # Trajectory
        if args_parser.show_trajectory:
            for track in online_targets:
                track_id = track.track_id
                tlwh = track.tlwh
                center_x = tlwh[0] + tlwh[2] / 2
                center_y = tlwh[1] + tlwh[3] / 2
                track_history[track_id].append((int(center_x), int(center_y)))

                if len(track_history[track_id]) > 50:
                    track_history[track_id].pop(0)

            for track_id, points in track_history.items():
                if len(points) > 1:
                    emotion = track_emotions.get(track_id, ("unknown", 0.0))[0]
                    color = get_emotion_color(emotion)
                    points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points_array], isClosed=False,
                                  color=color, thickness=2, lineType=cv2.LINE_AA)

        # Visualize với emotion
        frame = visualize_tracking_with_emotion(frame, online_targets, emotion_classifier,
                                                track_emotions, fps)

        # Hiển thị frame
        cv2.imshow("Face Tracking + Emotion Recognition - Webcam", frame)

        # Lưu video nếu cần
        if args_parser.save:
            out.write(frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"screenshot_{frame_id}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved: {screenshot_name}")

    # Giải phóng tài nguyên
    cap.release()
    if args_parser.save:
        out.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_id} frames")
    print("Done!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--webcam":
        sys.argv.pop(1)
        run_webcam()
    else:
        main()