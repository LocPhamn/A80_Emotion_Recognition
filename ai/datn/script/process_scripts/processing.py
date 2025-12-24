import glob
import shutil

import cv2
import os
import random

import numpy as np
from insightface.app import FaceAnalysis
from torch.utils.tensorboard.summary import image
from tqdm import tqdm
from deepface import DeepFace
import pandas as pd
from sklearn.cluster import DBSCAN


def convert_bbox_to_yolo(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    
    # Calculate center coordinates
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def save_yolo_annotation(faces, img_width, img_height, output_path, class_id=0):
    # Create test_images_yolo_format directory if it doesn't exist
    labels_dir = os.path.dirname(output_path)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    with open(output_path, 'w') as f:
        for face in faces:
            bbox = face.bbox.astype(int)
            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox, img_width, img_height
            )

            # Write in YOLO format: class_id x_center y_center width height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def save_frame():
    """ trích xuất khung hình từ video và lưu dưới dạng ảnh """
    root = r"D:\Python plus\AI_For_CV\script\datn\raw_data\train_videoss"
    out_dir = r"E:\data\train_images"
    video_paths = [os.path.join(root, v) for v in os.listdir(root)]

    for video_path in video_paths:
        base_name = os.path.basename(video_path).split('.')[0]
        image_folder = os.path.join(out_dir, base_name)
        os.makedirs(image_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {video_path}, total frames: {video_length}")
        frame_id = 0
        num_image = video_length

        while frame_id < num_image:
            ret, frame = cap.read()
            if not ret:
                break
            frame_name = f"frame_{frame_id}.jpg"
            cv2.imwrite(os.path.join(image_folder, frame_name), frame)
            frame_id += 1
        print(f"Extracted total {frame_id} frames from {video_path}")

def visualize_annotations():
    """ Hiển thị ảnh với bounding box từ file label YOLO """
    image_folder = "test_images_yolo_format/train_images"
    label_folder = "test_images_yolo_format/labels"

    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                   img.endswith(('.jpg', '.png', '.jpeg'))]

    for img_path in image_paths:
        base_name = os.path.basename(img_path).split('.')[0]
        label_path = os.path.join(label_folder, f"{base_name}.txt")

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {img_path}")
            continue

        h, w, _ = img.shape

        # Kiểm tra nếu không có file label
        if not os.path.exists(label_path):
            print(f"Không có label cho ảnh: {base_name}")
            continue

        # --- Đọc file YOLO label ---
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # bỏ qua dòng lỗi

            class_id = int(parts[0])
            x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])

            # Chuyển từ YOLO (normalized) -> pixel
            x1 = int((x_center - bbox_w / 2) * w)
            y1 = int((y_center - bbox_h / 2) * h)
            x2 = int((x_center + bbox_w / 2) * w)
            y2 = int((y_center + bbox_h / 2) * h)

            # --- Vẽ bounding box ---
            color = (0, 255, 0)  # màu xanh lá
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"ID:{class_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Hiển thị ảnh ---
        cv2.imshow("YOLO Annotation Viewer", img)
        key = cv2.waitKey(0)  # nhấn phím bất kỳ để sang ảnh tiếp theo
        if key == 27:  # ESC -> thoát
            break
    cv2.destroyAllWindows()

def resize_face(image):
    image = DeepFace.preprocessing.resize_image(image, (224, 224))
    return image

def size_processing(root, out_folder):
    """
    xử lý kích thước ảnh bằng deepface
    """

    images = [os.path.join(root, f) for f in os.listdir(root)]
    for path in images:
        image = cv2.imread(path)
        if image is None:
            print(f"Không thể đọc ảnh: {path}")
            continue

        resized_img = resize_face(image)[0] * 255
        resized_img = resized_img.astype('uint8')

        if resized_img is None or resized_img.size == 0:
            print(f"Lỗi xử lý ảnh: {path}")
            continue

        # lưu ảnh mới
        if out_folder is None:
            save_path = path
        else:
            save_path = os.path.join(out_folder,path)
        cv2.imwrite(save_path, resized_img)

    print("Hoàn tất xử lý và lưu ảnh!")

def label_emotions():
    """
    Đi qua từng ảnh, phân tích cảm xúc bằng DeepFace,
    và lưu lại với tên file là emotion_index.jpg
    """
    root = r"D:\Python plus\AI_For_CV\script\datn\data\test_faces"
    emotions = ['happy', 'angry', 'fear','surprise', 'sad', 'disgust','neutral']
    train_folder = r'D:\Python plus\AI_For_CV\script\datn\dataset\test'
    shutil.rmtree(train_folder)
    os.makedirs(train_folder, exist_ok=True)

    for emotion in emotions:
        os.makedirs(os.path.join(train_folder, emotion), exist_ok=True)

    # Lấy danh sách ảnh đã xử lý
    image_files = [f for f in os.listdir(root) if f.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Tìm thấy {len(image_files)} ảnh để phân tích...")


    for idx, img_name in enumerate(tqdm(image_files, desc="Analyzing emotions")):
        img_path = os.path.join(root, img_name)
        base_name = os.path.basename(img_name).split('.')[0]
        image = cv2.imread(img_path)
        resized_img = resize_face(image)[0] * 255
        resized_img = resized_img.astype('uint8')

        try:
            # Phân tích cảm xúc
            analysis = DeepFace.analyze(resized_img, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']

            # Lưu ảnh với tên emotion_index.jpg
            save_name = f"{train_folder}/{dominant_emotion}/{base_name}.jpg"
            cv2.imwrite(save_name, resized_img)

        except Exception as e:
            print(f"Lỗi phân tích ảnh {img_name}: {e}")
            continue
    print("Hoàn tất phân loại cảm xúc!")

def correct_label(label):
  if label == 'anger':
    return 'angry'
  elif label == 'happiness' or label == 'hapiness':
    return 'happy'
  elif label == 'surprise' or label == 'surpris':
    return 'surprise'
  elif label == 'neutral':
    return 'neutral'
  elif label == 'sadness':
    return 'sad'
  else:
    return label

def detect_and_crop_faces_deepface(root, face_dir, confidence_threshold=0.8):
    """
    Sử dụng DeepFace để phát hiện khuôn mặt, crop những khuôn mặt có confidence > threshold
    
    Args:
        image_paths: Danh sách đường dẫn ảnh cần xử lý
        face_dir: Thư mục lưu khuôn mặt đã crop
        confidence_threshold: Ngưỡng confidence (mặc định 0.8)
    """

    os.makedirs(face_dir, exist_ok=True)
    image_paths = [os.path.join(root, img) for img in os.listdir(root) if img.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Đang xử lý {len(image_paths)} ảnh...")
    
    face_count = 0
    total_faces = 0
    failed_images = 0
    sample = random.sample(image_paths, len(image_paths))
    for img_path in tqdm(sample, desc="Detecting and cropping faces"):
        try:
            # Đọc ảnh trước
            image = cv2.imread(img_path)
            if image is None:
                print(f"\nKhông thể đọc ảnh: {img_path}")
                failed_images += 1
                continue
            
            # Phát hiện khuôn mặt bằng DeepFace với backend opencv (ổn định hơn)
            try:
                faces = DeepFace.extract_faces(
                    img_path=image,  # Truyền numpy array thay vì path
                    detector_backend='opencv',  # Dùng opencv thay vì retinaface
                    enforce_detection=False,
                    align=False
                )
            except:
                # Thử backend khác nếu opencv fail
                faces = DeepFace.extract_faces(
                    img_path=image,
                    detector_backend='mtcnn',
                    enforce_detection=False,
                    align=False
                )
            
            for idx, face_obj in enumerate(faces):
                confidence = face_obj.get('confidence', 0)
                total_faces += 1
                
                # Chỉ lưu khuôn mặt có confidence > threshold
                if confidence > confidence_threshold:
                    # Lấy vùng khuôn mặt
                    facial_area = face_obj['facial_area']
                    x = facial_area['x']
                    y = facial_area['y']
                    w = facial_area['w']
                    h = facial_area['h']
                    
                    # Crop khuôn mặt
                    face_img = image[y:y+h, x:x+w]
                    
                    # Kiểm tra kích thước hợp lệ
                    if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                        continue
                    
                    # Lưu ảnh
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_name = f"{base_name}_face{idx}.jpg"
                    save_path = os.path.join(face_dir, save_name)
                    cv2.imwrite(save_path, face_img)
                    face_count += 1
                    
        except Exception as e:
            print(f"\nLỗi xử lý ảnh {os.path.basename(img_path)}: {str(e)[:100]}")
            failed_images += 1
            continue
    
    print(f"\n{'=' * 60}")
    print(f"Hoàn thành!")
    print(f"Tổng số khuôn mặt phát hiện: {total_faces}")
    print(f"Số khuôn mặt đạt confidence > {confidence_threshold}: {face_count}")
    print(f"Số ảnh lỗi: {failed_images}")
    print(f"Khuôn mặt đã lưu tại: {face_dir}")
    print(f"{'=' * 60}")

# trích xuất khuôn măt từ nửa sau video trung quốc
def extract_china_dynamic_face():
    root = r"C:\Users\ADMIN\Downloads\Dynamic_faces\Dynamic_faces"
    folders = [os.path.join(root, f) for f in os.listdir(root)]
    shutil.rmtree("D:\Python plus\AI_For_CV\script\datn\data\china_face")
    os.makedirs("D:\Python plus\AI_For_CV\script\datn\data\china_face", exist_ok=True)

    names = set()
    frame_id = 0
    for folder in folders:
        video_paths = [os.path.join(folder, img) for img in os.listdir(folder)]
        for video_path in video_paths:
            base_name = os.path.basename(video_path).split('_')[0].lower()
            if base_name not in names:
                os.makedirs(os.path.join("D:\Python plus\AI_For_CV\script\datn\data\china_face", base_name), exist_ok=True)
            names.add(base_name)

            cap = cv2.VideoCapture(video_path)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing video: {video_path}, total frames: {video_length}")
            start_frame = int(video_length * 0.7)

            for i in range(0, video_length):
                ret, frame = cap.read()
                if not ret:
                    break
                if i < start_frame:
                    continue
                frame_name = os.path.join(f"D:\Python plus\AI_For_CV\script\datn\data\china_face", base_name) + f"/frame_{frame_id:04d}.jpg"
                cv2.imwrite(frame_name, frame)
                frame_id += 1

            print(f"Extracted total {frame_id} frames from {video_path}")
    print(names)

def get_embedding(image_path, model='VGG-Face'):
    """
    Trích xuất embedding từ ảnh (chỉ tính 1 lần)
    """
    try:
        result = DeepFace.represent(image_path,
                                    model_name=model,
                                    enforce_detection=False)
        return np.array(result[0]['embedding'])
    except Exception as e:
        print(f"Lỗi embedding {image_path}: {e}")
        return None

def cosine_similarity(emb1, emb2):
    """
    Tính cosine similarity giữa 2 embeddings
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def similarity_filter_optimized(root, threshold=0.4, model='VGG-Face', log_file="image_similarity_log_test_img.csv"):
    """
    Lọc ảnh dựa trên embedding - Nhanh hơn nhiều!
    threshold: < threshold thì giữ lại (với cosine distance)
    """

    # Kiểm tra file CSV có tồn tại không
    file_exists = os.path.exists(log_file)

    # Tạo DataFrame mới cho root này
    data_frame = pd.DataFrame(columns=["root", "image_path", "min_similarity", "is_unique"])

    images = sorted([os.path.join(root, f) for f in os.listdir(root)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    unique_embeddings = []
    unique_images = []

    print(f"Đang xử lý thư mục: {root}")
    print(f"Tổng số ảnh: {len(images)}")

    for idx, img_path in enumerate(tqdm(images, desc=f"Processing {os.path.basename(root)}")):
        current_emb = get_embedding(img_path, model)

        if current_emb is None:
            continue

        if len(unique_embeddings) == 0:
            unique_embeddings.append(current_emb)
            unique_images.append(img_path)
            data_frame.loc[len(data_frame)] = [root, img_path, 0.0, True]
        else:
            similarities = [1 - cosine_similarity(current_emb, saved_emb)
                            for saved_emb in unique_embeddings]
            min_distance = min(similarities)

            if min_distance > threshold:
                unique_embeddings.append(current_emb)
                unique_images.append(img_path)
                is_unique = True
            else:
                is_unique = False

            data_frame.loc[len(data_frame)] = [root, img_path, min_distance, is_unique]

    if file_exists:
        data_frame.to_csv(log_file, mode='a', header=False, index=False)
    else:
        data_frame.to_csv(log_file, mode='w', header=True, index=False)
    print(f"✅ Thư mục {os.path.basename(root)}: {len(unique_images)}/{len(images)} ảnh unique")
    return unique_images


def similarity_filter_clustering(root, threshold=0.4, model='VGG-Face', method='dbscan'):
    """
    Lọc ảnh bằng clustering - Nhanh hơn nhiều!

    Args:
        root: Thư mục chứa ảnh
        threshold: Ngưỡng similarity (0-1)
        model: Model embedding
        method: 'dbscan' hoặc 'agglomerative'

    Returns:
        unique_images: Danh sách ảnh đại diện
    """
    log_image = "image_clustering_log.csv"

    # Lấy danh sách ảnh
    images = sorted([os.path.join(root, f) for f in os.listdir(root)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    print(f"Bắt đầu trích xuất embedding cho {len(images)} ảnh...")

    # Bước 1: Trích xuất TẤT CẢ embeddings
    embeddings = []
    valid_images = []

    for img_path in tqdm(images):
        emb = get_embedding(img_path, model)
        if emb is not None:
            embeddings.append(emb)
            valid_images.append(img_path)

    embeddings = np.array(embeddings)
    print(f"Đã trích xuất {len(embeddings)} embeddings")

    # Bước 2: Clustering
    print(f"Đang clustering với threshold={threshold}...")

    if method == 'dbscan':
        # DBSCAN: tự động tìm số cluster
        # eps = threshold vì ta dùng cosine distance
        clustering = DBSCAN(eps=threshold, min_samples=1, metric='cosine')
        labels = clustering.fit_predict(embeddings)
    else:
        # Agglomerative Clustering (nếu muốn kiểm soát số cluster)
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage='average',
            metric='cosine'
        )
        labels = clustering.fit_predict(embeddings)

    # Bước 3: Chọn đại diện cho mỗi cluster
    unique_images = []
    cluster_info = []

    for cluster_id in np.unique(labels):
        # Lấy tất cả ảnh trong cluster này
        cluster_mask = (labels == cluster_id)
        cluster_images = [valid_images[i] for i in np.where(cluster_mask)[0]]
        cluster_embeddings = embeddings[cluster_mask]

        # Chọn ảnh đại diện: ảnh gần centroid nhất
        centroid = cluster_embeddings.mean(axis=0)
        distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
        representative_idx = np.argmin(distances)
        representative_image = cluster_images[representative_idx]

        unique_images.append(representative_image)

        # Lưu thông tin cluster
        for img in cluster_images:
            cluster_info.append({
                'image_path': img,
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_images),
                'is_representative': (img == representative_image)
            })

    # Lưu log
    df = pd.DataFrame(cluster_info)
    df.to_csv(log_image, index=False)

    print(f"\n=== KẾT QUẢ ===")
    print(f"Tổng số ảnh: {len(images)}")
    print(f"Số cluster: {len(np.unique(labels))}")
    print(f"Ảnh đại diện: {len(unique_images)}")
    print(f"Đã loại bỏ: {len(valid_images) - len(unique_images)}")
    print(f"Log lưu tại: {log_image}")

    return unique_images


if __name__ == '__main__':
    # label_emotions()
    # extract_china_dynamic_face()
    # video_to_face_image(r"D:\Python plus\AI_For_CV\script\datn\data\train_videoss")
    # root = r"E:\data\test_images\*faces"
    # folders = [os.path.join(root, f) for f in os.listdir(root)]
    # folders = glob.glob(root)
    # folders.reverse()
    # for folder in folders:
        # detect_and_crop_faces_deepface(folder,folder+"faces", confidence_threshold=0.9)
        # size_processing(folder+"faces", None)
        # ========================================

        # uni_image = similarity_filter_optimized(folder, threshold=0.4, model='VGG-Face')
        # uni_image = similarity_filter_clustering(folder, threshold=0.4, model='VGG-Face', method='dbscan')


    # detect_and_crop_faces_deepface(r"D:\Python plus\AI_For_CV\script\datn\raw_data\test_images\IMG_5252",r"D:\Python plus\AI_For_CV\script\datn\data\test_face", confidence_threshold=0.9)
    # size_processing(r"D:\Python plus\AI_For_CV\script\datn\data\test_face",None)

    df = pd.read_csv("image_similarity_log_test_img.csv")
    os.mkdir(r"E:\data\un_duplicate_test_faces")
    unique_image_paths = df.loc[df["is_unique"] == True, "image_path"]
    for idx,img_path in enumerate(unique_image_paths):
        base_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(r"E:\data\un_duplicate_test_faces", base_name))
        # cv2.imshow("Unique Image", cv2.imread(img_path))
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #
        # if cv2.waitKey(0) & 0xFF == 27:
        #     break