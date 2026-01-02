import os
import random
import glob
import cv2
from deepface import DeepFace
from tqdm import tqdm

from processing import resize_face


def detect_and_crop_faces_deepface(root, face_dir, confidence_threshold=0.8):
    """
    Sử dụng DeepFace để phát hiện khuôn mặt, crop những khuôn mặt có confidence > threshold

    Args:
        image_paths: Danh sách đường dẫn ảnh cần xử lý
        face_dir: Thư mục lưu khuôn mặt đã crop
        confidence_threshold: Ngưỡng confidence (mặc định 0.8)
    """

    # os.makedirs(face_dir, exist_ok=True)
    image_paths = root

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
                    face_img = image[y:y + h, x:x + w]

                    # Kiểm tra kích thước hợp lệ
                    if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                        continue

                    # Lưu ảnh
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_name = f"{base_name}_face{idx}.jpg"
                    save_path = os.path.join(img_path)
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

def size_processing(root, out_folder):
    """
    xử lý kích thước ảnh bằng deepface
    """

    images = root
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

def balance_datasets():
    """
    Thống kê và so sánh số lượng ảnh giữa 2 dataset AffectNet và A80
    """
    # ** P2: Cân bằng dữ liệu bộ A80
    target_number = 1000
    if len(common_images) > target_number:
        images_to_keep = set(random.sample(common_images, target_number))
        images_to_delete = set(common_images) - images_to_keep

        print(f"Xóa {len(images_to_delete)} ảnh để còn đúng {target_number}")
        for img in images_to_delete:
            img_path = os.path.join(a80_folder, img)
            if os.path.exists(img_path):
                os.remove(img_path)
    else:
        print(f"Không cần xóa, số ảnh giao nhỏ hơn hoặc bằng {target_number}")

if __name__ == '__main__':
    root = r"E:\archive (3)"

    affectnet_train = os.path.join(root,"Train_AffectNet")
    a80_train = os.path.join(root,r"dataset\trainv2")

    emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry']

    for emotion in emotions:
        # ** P1: Thống kê số lượng ảnh trong từng folder
        tmp = emotion
        if emotion == 'angry':
            tmp = 'anger'
        affectnet_folder = os.path.join(affectnet_train, tmp)
        a80_folder = os.path.join(a80_train, emotion)

        affectnet_imagess = set(os.listdir(affectnet_folder))
        a80_images = set(os.listdir(a80_folder))
        a80_image_paths = glob.glob(os.path.join(a80_folder, "*"))
        # print(f"{emotion}: AffectNet={len(affectnet_imagess)}, A80={len(a80_images)}")

        # print("Anh giao nhau giua 2 dataset:")
        # common_images = affectnet_imagess.intersection(a80_images)
        # print(f"Emotion {emotion}: {len(common_images)} anh giao nhau")
        # print("="*50)

        # ** P2: thống kê dữ liệu trung quốc va A80 trong bộ dữ liệu
        china_end = "china*.jpg"
        a80_end = f"{emotion}*.jpg"

        china_images = glob.glob(os.path.join(a80_folder, china_end))
        a80_emotion_images = glob.glob(os.path.join(a80_folder, a80_end))

        affect_net_images = set(a80_image_paths) - set(china_images) - set(a80_emotion_images)
        # detect_and_crop_faces_deepface(affect_net_images, affectnet_folder)
        # size_processing(affect_net_images, None)
        print(f"ảnh trung quốc {emotion}: ", len(china_images))
        print(f"ảnh A80 emotion {emotion}: ", len(a80_emotion_images))
        print("tổng ảnh: ", len(a80_images))
        print("="*50)

if __name__ == '__main__':
    pass
