import os
import shutil

from PIL import Image
import pillow_heif
import cv2
import random
import numpy as np

def convert_heic_to_jpg(heic_file_path, jpg_file_path):
    try:
        # Đăng ký heif opener cho PIL
        pillow_heif.register_heif_opener()

        # Mở ảnh bằng PIL
        img = Image.open(heic_file_path)

        # Convert sang RGB nếu cần
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Lưu dưới dạng JPEG
        img.save(jpg_file_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Lỗi: {e}")
        return False
def convert_images():
    root = r"C:\Users\ADMIN\Downloads\Ảnh cảm xúc-20251203T182517Z-1-001\Ảnh cảm xúc"
    images = [os.path.join(root, f) for f in os.listdir(root)]

    out_folder = r"D:\Python plus\AI_For_CV\script\datn\data\heic_to_jpg"
    os.makedirs(out_folder, exist_ok=True)

    for image in images:
        if image.lower().endswith('.heic'):
            base_name = os.path.basename(image).lower()
            output_jpg = os.path.join(out_folder, base_name.replace('.heic', '.jpg'))
            if convert_heic_to_jpg(image, output_jpg):
                print(f"✓ Converted successfully: {output_jpg}")
            else:
                print(f"✗ Conversion failed for: {image}")
def advanced_rotation_augmentation(image, rotation_type='random'):
    """
    Xoay ảnh khuôn mặt với nhiều góc độ và kỹ thuật khác nhau
    rotation_type: 'small', 'medium', 'large', '3d_left', '3d_right', '3d_up', '3d_down', 'random'
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    if rotation_type == 'random':
        rotation_type = random.choice(['small', 'medium', 'large', '3d_left', '3d_right', '3d_up', '3d_down'])
    
    # 1. Xoay góc nhỏ (-10 đến 10 độ)
    if rotation_type == 'small':
        angle = random.uniform(-10, 10)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 2. Xoay góc trung bình (-25 đến 25 độ)
    elif rotation_type == 'medium':
        angle = random.uniform(-25, 25)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 3. Xoay góc lớn (-45 đến 45 độ)
    elif rotation_type == 'large':
        angle = random.uniform(-30, 30)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 4-7. Xoay 3D perspective (nhẹ hơn, tự nhiên hơn)
    elif rotation_type in ['3d_left', '3d_right', '3d_up', '3d_down']:
        # Góc xoay nhẹ cho khuôn mặt
        intensity = random.uniform(0.05, 0.1)  # 2-5% thay vì 5-10%
        
        if rotation_type == '3d_left':
            src_pts = np.float32([
                [0, 0],
                [w, int(h * intensity)],
                [0, h],
                [w, h - int(h * intensity)]
            ])
        elif rotation_type == '3d_right':
            src_pts = np.float32([
                [0, int(h * intensity)],
                [w, 0],
                [0, h - int(h * intensity)],
                [w, h]
            ])
        elif rotation_type == '3d_up':
            src_pts = np.float32([
                [0, 0],
                [w, 0],
                [int(w * intensity), h],
                [w - int(w * intensity), h]
            ])
        else:  # 3d_down
            src_pts = np.float32([
                [int(w * intensity), 0],
                [w - int(w * intensity), 0],
                [0, h],
                [w, h]
            ])
        
        dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    return image


def combined_rotation_augmentation(image, num_rotations=3):
    """
    Tạo nhiều ảnh xoay từ 1 ảnh gốc
    Trả về list các ảnh đã xoay
    """
    rotated_images = []
    
    rotation_types = ['small', 'medium', 'large', '3d_left', '3d_right', '3d_up', '3d_down']
    
    # Random chọn các loại xoay
    selected_types = random.sample(rotation_types, min(num_rotations, len(rotation_types)))
    
    for rot_type in selected_types:
        rotated = advanced_rotation_augmentation(image, rotation_type=rot_type)
        rotated_images.append((rotated, rot_type))
    
    return rotated_images


def generate_rotated_dataset(input_folder, output_folder, rotations_per_image=5):
    """
    Tạo dataset với nhiều góc xoay
    """
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)
    emotions = os.listdir(input_folder)
    total_generated = 0
    
    for emotion in emotions:
        emotion_path = os.path.join(input_folder, emotion)
        if not os.path.isdir(emotion_path):
            continue
        
        output_emotion_path = os.path.join(output_folder, emotion)
        os.makedirs(output_emotion_path, exist_ok=True)

        
        images = [f for f in os.listdir(emotion_path) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_name in images:
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            base_name = os.path.splitext(img_name)[0]
            
            # Lưu ảnh gốc
            cv2.imwrite(os.path.join(output_emotion_path, f"{base_name}_original.jpg"), img)
            total_generated += 1
            
            # Tạo các góc xoay
            rotation_types = ['small', 'medium', 'large', '3d_left', '3d_right', '3d_up', '3d_down']
            
            for i in range(rotations_per_image):
                rot_type = random.choice(rotation_types)
                rotated = advanced_rotation_augmentation(img, rotation_type=rot_type)
                
                output_name = f"{base_name}_rot_{rot_type}_{i+1}.jpg"
                cv2.imwrite(os.path.join(output_emotion_path, output_name), rotated)
                total_generated += 1
            
            print(f"✓ Generated {rotations_per_image + 1} images for {emotion}/{img_name}")
    
    print(f"\n✓ Total images generated: {total_generated}")


if __name__ == "__main__":
    generate_rotated_dataset("D:\Python plus\AI_For_CV\script\datn\data\internet_face","D:\Python plus\AI_For_CV\script\datn\data\gen_images_v2", rotations_per_image=10)
    