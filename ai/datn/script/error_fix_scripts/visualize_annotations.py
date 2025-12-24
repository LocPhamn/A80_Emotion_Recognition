import cv2
import os
import numpy as np
from pathlib import Path

def read_yolo_annotations(label_path):
    """
    Đọc file annotation YOLO format
    Returns: List of bounding boxes [(class_id, x_center, y_center, width, height), ...]
    """
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append((class_id, x_center, y_center, width, height))
    return annotations

def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    Chuyển đổi từ YOLO format sang tọa độ bounding box thực tế
    """
    # Chuyển từ normalized coordinates sang pixel coordinates
    x_center_pixel = x_center * img_width
    y_center_pixel = y_center * img_height
    width_pixel = width * img_width
    height_pixel = height * img_height
    
    # Tính toán tọa độ góc trái trên và góc phải dưới
    x1 = int(x_center_pixel - width_pixel / 2)
    y1 = int(y_center_pixel - height_pixel / 2)
    x2 = int(x_center_pixel + width_pixel / 2)
    y2 = int(y_center_pixel + height_pixel / 2)
    
    return x1, y1, x2, y2

def draw_bounding_boxes(image, annotations):
    """
    Vẽ bounding boxes lên ảnh
    """
    img_height, img_width = image.shape[:2]
    
    # Các màu khác nhau cho các class
    colors = [
        (0, 255, 0),    # Xanh lá cho class 0 (face)
        (255, 0, 0),    # Xanh dương cho class 1
        (0, 0, 255),    # Đỏ cho class 2
        (255, 255, 0),  # Cyan cho class 3
        (255, 0, 255),  # Magenta cho class 4
    ]
    
    for class_id, x_center, y_center, width, height in annotations:
        # Chuyển đổi sang tọa độ pixel
        x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
        
        # Chọn màu dựa trên class_id
        color = colors[class_id % len(colors)]
        
        # Vẽ rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ label
        label = f"Class {class_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

def visualize_single_image(image_path, label_path, output_path=None, show=True):
    """
    Hiển thị một ảnh với bounding boxes
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    # Đọc annotations
    annotations = read_yolo_annotations(label_path)
    if not annotations:
        print(f"Không có annotation trong file: {label_path}")
    
    # Vẽ bounding boxes
    result_image = draw_bounding_boxes(image.copy(), annotations)
    
    # Lưu ảnh kết quả nếu có đường dẫn output
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Đã lưu ảnh kết quả tại: {output_path}")
    
    # Hiển thị ảnh
    if show:
        window_name = f"Image: {os.path.basename(image_path)}"
        cv2.imshow(window_name, result_image)
        print(f"Nhấn phím bất kỳ để tiếp tục... (Đang hiển thị: {os.path.basename(image_path)})")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_image

def visualize_batch(image_folder, label_folder, output_folder=None, show_each=False):
    """
    Hiển thị tất cả ảnh trong thư mục với bounding boxes
    """
    # Tạo thư mục output nếu cần
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Lấy danh sách tất cả ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f'*{ext}'))
        image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"Không tìm thấy ảnh nào trong thư mục: {image_folder}")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    for image_file in image_files:
        image_path = str(image_file)
        base_name = image_file.stem
        label_path = os.path.join(label_folder, f"{base_name}.txt")
        
        if not os.path.exists(label_path):
            print(f"Không tìm thấy label file cho: {image_path}")
            continue
        
        # Đường dẫn output
        output_path = None
        if output_folder:
            output_path = os.path.join(output_folder, f"{base_name}_annotated.jpg")
        
        # Hiển thị ảnh
        visualize_single_image(image_path, label_path, output_path, show_each)

def main():
    """
    Hàm chính - có thể chỉnh sửa để sử dụng
    """
    # Cấu hình đường dẫn
    image_folder = "test_images_yolo_format/train_images"
    label_folder = "test_images_yolo_format/labels"
    output_folder = "output_visualized"  # Có thể để None nếu không muốn lưu
    
    print("=== CHƯƠNG TRÌNH HIỂN THỊ BOUNDING BOX ===")
    print("1. Hiển thị một ảnh cụ thể")
    print("2. Hiển thị tất cả ảnh trong thư mục")
    print("3. Xử lý batch và lưu kết quả")
    
    choice = input("Chọn chế độ (1/2/3): ").strip()
    
    if choice == "1":
        # Hiển thị một ảnh cụ thể
        image_name = input("Nhập tên ảnh (vd: frame0.jpg): ").strip()
        image_path = os.path.join(image_folder, image_name)
        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(label_folder, f"{base_name}.txt")
        
        visualize_single_image(image_path, label_path)
        
    elif choice == "2":
        # Hiển thị tất cả ảnh lần lượt
        visualize_batch(image_folder, label_folder, show_each=True)
        
    elif choice == "3":
        # Xử lý batch và lưu kết quả
        visualize_batch(image_folder, label_folder, output_folder, show_each=False)
        print(f"Đã xử lý xong! Kết quả được lưu trong thư mục: {output_folder}")
    
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()