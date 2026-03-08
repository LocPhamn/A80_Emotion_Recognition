import cv2
import numpy as np
import os
import glob
from pathlib import Path
import random
import matplotlib.pyplot as plt


def calculate_average_v_channel_all_folders(parent_folder_path):
    """
    Tính kênh V trung bình của tất cả ảnh trong TẤT CẢ các folder con

    Args:
        parent_folder_path (str): Đường dẫn đến folder cha chứa 7 folder cảm xúc

    Returns:
        float: Giá trị trung bình của kênh V
        dict: Thống kê chi tiết
    """
    # Lấy danh sách tất cả folder con
    emotion_folders = [f for f in os.listdir(parent_folder_path)
                       if os.path.isdir(os.path.join(parent_folder_path, f))]

    if not emotion_folders:
        print(f"❌ Không tìm thấy folder con nào trong: {parent_folder_path}")
        return None, None

    print(f"Tìm thấy {len(emotion_folders)} folder cảm xúc: {', '.join(emotion_folders)}")
    print("=" * 80)

    # Các định dạng ảnh được hỗ trợ
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']

    # Thu thập tất cả ảnh từ tất cả các folder
    all_image_files = []
    for emotion_name in emotion_folders:
        emotion_path = os.path.join(parent_folder_path, emotion_name)
        for ext in image_extensions:
            all_image_files.extend(glob.glob(os.path.join(emotion_path, ext)))
            all_image_files.extend(glob.glob(os.path.join(emotion_path, ext.upper())))

    if not all_image_files:
        print(f"❌ Không tìm thấy ảnh nào trong các folder con!")
        return None, None

    print(f"\nTổng số ảnh tìm thấy: {len(all_image_files)}")
    print("\nĐang xử lý tất cả ảnh...")
    print("-" * 80)

    all_v_values = []
    v_means_per_image = []
    successful_images = 0
    failed_images = []

    # Xử lý tất cả ảnh
    for i, img_path in enumerate(all_image_files, 1):
        try:
            # Đọc ảnh
            img = cv2.imread(img_path)

            if img is None:
                failed_images.append(Path(img_path).name)
                continue

            # Convert sang HSV
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Lấy kênh V (Value) - kênh thứ 3 (index 2)
            v_channel = hsv_img[:, :, 2]

            # Tính giá trị trung bình của kênh V cho ảnh này
            v_mean = np.mean(v_channel)
            v_means_per_image.append(v_mean)

            # Lưu tất cả giá trị V để tính trung bình tổng thể
            all_v_values.extend(v_channel.flatten())

            successful_images += 1

            # Hiển thị tiến trình mỗi 100 ảnh
            if i % 100 == 0:
                print(f"   Đã xử lý: {i}/{len(all_image_files)} ảnh...")

        except Exception as e:
            failed_images.append(Path(img_path).name)

    if not all_v_values:
        print("\n❌ Không có ảnh nào được xử lý thành công!")
        return None, None

    print(f"   Hoàn thành: {successful_images}/{len(all_image_files)} ảnh")

    # Tính kênh V trung bình
    average_v = np.mean(all_v_values)
    average_v_per_image = np.mean(v_means_per_image)

    # Thống kê
    stats = {
        'total_images': len(all_image_files),
        'successful_images': successful_images,
        'failed_images': len(failed_images),
        'failed_list': failed_images,
        'average_v_all_pixels': average_v,
        'average_v_per_image': average_v_per_image,
        'min_v_per_image': np.min(v_means_per_image) if v_means_per_image else 0,
        'max_v_per_image': np.max(v_means_per_image) if v_means_per_image else 0,
        'std_v_per_image': np.std(v_means_per_image) if v_means_per_image else 0
    }

    return average_v, stats


def print_statistics(average_v, stats):
    """In kết quả thống kê"""
    if average_v is None or stats is None:
        return

    print("\n" + "=" * 80)
    print("KẾT QUẢ THỐNG KÊ KÊNH V TRUNG BÌNH (TẤT CẢ 7 FOLDER CẢM XÚC)")
    print("=" * 80)
    print(f"Tổng số ảnh tìm thấy: {stats['total_images']}")
    print(f"Số ảnh xử lý thành công: {stats['successful_images']}")
    print(f"Số ảnh thất bại: {stats['failed_images']}")

    if stats['failed_list'] and len(stats['failed_list']) <= 10:
        print(f"\nDanh sách ảnh thất bại:")
        for fname in stats['failed_list']:
            print(f"  - {fname}")
    elif stats['failed_list']:
        print(f"\nCó {len(stats['failed_list'])} ảnh thất bại (hiển thị 10 đầu tiên):")
        for fname in stats['failed_list'][:10]:
            print(f"  - {fname}")

    print("\n" + "-" * 80)
    print("GIÁ TRỊ KÊNH V (Value in HSV)")
    print("-" * 80)
    print(f"Kênh V trung bình (tất cả pixels): {average_v:.2f}")
    print(f"Kênh V trung bình (theo ảnh): {stats['average_v_per_image']:.2f}")
    print(f"Kênh V min (theo ảnh): {stats['min_v_per_image']:.2f}")
    print(f"Kênh V max (theo ảnh): {stats['max_v_per_image']:.2f}")
    print(f"Độ lệch chuẩn (theo ảnh): {stats['std_v_per_image']:.2f}")
    print("=" * 80)


def display_extreme_v_images(parent_folder_path, num_images=10):
    """
    Tìm và hiển thị 10 ảnh có V min và 10 ảnh có V max

    Args:
        parent_folder_path (str): Đường dẫn đến folder cha
        num_images (int): Số ảnh cần lấy cho mỗi loại (min/max)
    """
    print("\n" + "=" * 80)
    print("TÌM VÀ HIỂN THỊ ẢNH CÓ V MIN VÀ V MAX")
    print("=" * 80)
    print(f"Số ảnh cho mỗi loại: {num_images}")

    # Lấy danh sách tất cả folder con
    emotion_folders = [f for f in os.listdir(parent_folder_path)
                       if os.path.isdir(os.path.join(parent_folder_path, f))]

    # Các định dạng ảnh được hỗ trợ
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']

    # Thu thập tất cả ảnh từ tất cả các folder
    all_image_files = []
    for emotion_name in emotion_folders:
        emotion_path = os.path.join(parent_folder_path, emotion_name)
        for ext in image_extensions:
            all_image_files.extend(glob.glob(os.path.join(emotion_path, ext)))
            all_image_files.extend(glob.glob(os.path.join(emotion_path, ext.upper())))

    if not all_image_files:
        print("❌ Không tìm thấy ảnh nào!")
        return

    print(f"Tổng số ảnh: {len(all_image_files)}")
    print("Đang tính toán giá trị V cho tất cả ảnh...")

    # Tính giá trị V cho tất cả ảnh
    image_v_list = []
    for img_path in all_image_files:
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert sang HSV
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Lấy kênh V và tính mean
            v_channel = hsv_img[:, :, 2]
            v_mean = np.mean(v_channel)

            image_v_list.append((img_path, v_mean))

        except Exception as e:
            continue

    if len(image_v_list) < num_images * 2:
        print(f"⚠️ Chỉ tìm thấy {len(image_v_list)} ảnh hợp lệ")
        num_images = min(num_images, len(image_v_list) // 2)

    # Sắp xếp theo V value
    image_v_list.sort(key=lambda x: x[1])

    # Lấy 10 ảnh có V thấp nhất và 10 ảnh có V cao nhất
    min_v_images = image_v_list[:num_images]
    max_v_images = image_v_list[-num_images:]

    print(f"\n✓ Đã tìm thấy {num_images} ảnh V min và {num_images} ảnh V max")
    print(f"V min range: {min_v_images[0][1]:.2f} - {min_v_images[-1][1]:.2f}")
    print(f"V max range: {max_v_images[0][1]:.2f} - {max_v_images[-1][1]:.2f}")
    print("\n" + "-" * 80)

    # Tạo figure để hiển thị - 2 hàng (min và max)
    fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 7))

    # Hiển thị 10 ảnh V min
    print("\n📉 ẢNH CÓ V THẤP NHẤT:")
    for i, (img_path, v_mean) in enumerate(min_v_images):
        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[0, i].imshow(img_rgb)
            axes[0, i].set_title(f'V={v_mean:.2f}', fontsize=10)
            axes[0, i].axis('off')

            print(f"  [{i+1}] {Path(img_path).name}: V={v_mean:.2f}")

        except Exception as e:
            print(f"  [{i+1}] ❌ Lỗi: {Path(img_path).name}")

    # Hiển thị 10 ảnh V max
    print("\n📈 ẢNH CÓ V CAO NHẤT:")
    for i, (img_path, v_mean) in enumerate(max_v_images):
        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[1, i].imshow(img_rgb)
            axes[1, i].set_title(f'V={v_mean:.2f}', fontsize=10)
            axes[1, i].axis('off')

            print(f"  [{i+1}] {Path(img_path).name}: V={v_mean:.2f}")

        except Exception as e:
            print(f"  [{i+1}] ❌ Lỗi: {Path(img_path).name}")

    # Thêm labels cho từng hàng
    axes[0, 0].text(-0.1, 0.5, '10 ảnh V MIN',
                    transform=axes[0, 0].transAxes,
                    fontsize=14, fontweight='bold',
                    va='center', ha='right', rotation=90)

    axes[1, 0].text(-0.1, 0.5, '10 ảnh V MAX',
                    transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight='bold',
                    va='center', ha='right', rotation=90)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print("✓ HOÀN THÀNH! Đã hiển thị 10 ảnh V min và 10 ảnh V max")
    print("=" * 80)


if __name__ == "__main__":
    # Nhập đường dẫn folder
    print("=" * 80)
    print("CHƯƠNG TRÌNH TÍNH KÊNH V TRUNG BÌNH (TẤT CẢ 7 FOLDER CẢM XÚC)")
    print("=" * 80)

    folder_path = r"E:\archive (3)\dataset\trainv3 - Copy\trainv3 - Copy"

    # Loại bỏ dấu ngoặc kép nếu có
    folder_path = folder_path.strip('"').strip("'")

    # Kiểm tra folder có tồn tại không
    if not os.path.exists(folder_path):
        print(f"\n❌ Folder không tồn tại: {folder_path}")
        exit(1)

    if not os.path.isdir(folder_path):
        print(f"\n❌ Đường dẫn không phải là folder: {folder_path}")
        exit(1)

    print(f"\n✓ Folder hợp lệ: {folder_path}\n")

    # Tính kênh V trung bình của tất cả ảnh trong tất cả các folder
    # average_v, stats = calculate_average_v_channel_all_folders(folder_path)

    # In kết quả
    # print_statistics(average_v, stats)

    # Hiển thị 10 ảnh có V min và 10 ảnh có V max
    display_extreme_v_images(folder_path, num_images=10)
