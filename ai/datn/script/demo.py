import os
import time
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# --- Đường dẫn dataset (nếu cần dùng sau này) ---
root = r"D:\Python plus\AI_For_CV\dataset\datn_face_dataset\test"
image_folder = os.path.join(root, "train_images")
label_folder = os.path.join(root, "labels")

# --- Load model YOLO ---
model = YOLO("yolo_models/yolov11s_custom.pt")

# --- Mở camera ---
cap = cv2.VideoCapture("data/train_videoss/IMG_5279.MOV")

# Kiểm tra camera
if not cap.isOpened():
    exit()

fps = 0
prev_time = time.time()
# Store the track history
track_history = defaultdict(lambda: [])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Tính FPS ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # --- Chạy YOLO ---
    result = model.track(frame, persist=True,tracker="bytetrack.yaml")[0]

    # Visualize the results on the frame
    annotated_frame = result[0].plot()

    # Get the boxes and track IDs
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Visualize the result on the frame
        frame = result.plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    # Display the annotated frame
    cv2.imshow("YOLO11 Tracking", frame)
    # Display the annotated frame
    cv2.imshow("YOLO11 Tracking", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
