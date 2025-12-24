import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict
import shutil

def calculate_hashes(image_path):
    """
    Tính toán các loại hash cho một ảnh: pHash, aHash, dHash
    """
    try:
        # Đọc ảnh bằng PIL
        img = Image.open(image_path)
        
        # Tính các loại hash
        phash = imagehash.phash(img)
        ahash = imagehash.average_hash(img)
        dhash = imagehash.dhash(img)
        
        return {
            'phash': phash,
            'ahash': ahash,
            'dhash': dhash
        }
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """
    Tính khoảng cách Hamming giữa hai hash
    """
    return hash1 - hash2

def find_similar_images(folder_path, threshold=5):
    """
    Tìm các ảnh giống nhau trong folder dựa trên hash
    
    Args:
        folder_path: đường dẫn đến folder chứa ảnh
        threshold: ngưỡng khoảng cách Hamming (càng nhỏ càng giống)
    
    Returns:
        dict: dictionary chứa các nhóm ảnh giống nhau
    """
    print(f"Đang quét folder: {folder_path}")
    
    # Lấy danh sách tất cả file ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    # Tính hash cho tất cả ảnh
    image_hashes = {}
    for i, image_path in enumerate(image_files):
        print(f"Đang xử lý ảnh {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        hashes = calculate_hashes(image_path)
        if hashes:
            image_hashes[image_path] = hashes
    
    # Tìm các ảnh giống nhau
    similar_groups = []
    processed = set()
    
    for img1_path, hashes1 in image_hashes.items():
        if img1_path in processed:
            continue
            
        similar_group = [img1_path]
        processed.add(img1_path)
        
        for img2_path, hashes2 in image_hashes.items():
            if img2_path in processed:
                continue
                
            # So sánh các loại hash
            phash_dist = hamming_distance(hashes1['phash'], hashes2['phash'])
            ahash_dist = hamming_distance(hashes1['ahash'], hashes2['ahash'])
            dhash_dist = hamming_distance(hashes1['dhash'], hashes2['dhash'])
            
            # Nếu ít nhất 2 trong 3 loại hash có khoảng cách nhỏ hơn threshold
            similar_count = sum([
                phash_dist <= threshold,
                ahash_dist <= threshold,
                dhash_dist <= threshold
            ])
            
            if similar_count >= 2:
                similar_group.append(img2_path)
                processed.add(img2_path)
                print(f"Tìm thấy ảnh tương tự: {os.path.basename(img1_path)} và {os.path.basename(img2_path)}")
                print(f"  pHash: {phash_dist}, aHash: {ahash_dist}, dHash: {dhash_dist}")
        
        if len(similar_group) > 1:
            similar_groups.append(similar_group)
    
    return similar_groups

def remove_duplicate_images(similar_groups, keep_first=True, backup_folder=None):
    """
    Xóa các ảnh trùng lặp
    
    Args:
        similar_groups: danh sách các nhóm ảnh giống nhau
        keep_first: True nếu giữ ảnh đầu tiên, False nếu giữ ảnh cuối
        backup_folder: folder để backup ảnh trước khi xóa (tùy chọn)
    """
    total_removed = 0
    
    # Tạo folder backup nếu cần
    if backup_folder and not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
        print(f"Đã tạo folder backup: {backup_folder}")
    
    for i, group in enumerate(similar_groups):
        print(f"\nNhóm {i+1}: {len(group)} ảnh giống nhau")
        for img_path in group:
            print(f"  - {os.path.basename(img_path)}")
        
        # Sắp xếp theo tên file để có thứ tự nhất quán
        group.sort()
        
        # Chọn ảnh để giữ lại
        if keep_first:
            keep_image = group[0]
            remove_images = group[1:]
        else:
            keep_image = group[-1]
            remove_images = group[:-1]
        
        print(f"Giữ lại: {os.path.basename(keep_image)}")
        
        # Xóa các ảnh trùng lặp
        for img_path in remove_images:
            try:
                # Backup nếu cần
                if backup_folder:
                    backup_path = os.path.join(backup_folder, os.path.basename(img_path))
                    shutil.copy2(img_path, backup_path)
                    print(f"  Đã backup: {os.path.basename(img_path)}")
                
                # Xóa ảnh
                os.remove(img_path)
                print(f"  Đã xóa: {os.path.basename(img_path)}")
                total_removed += 1
                
            except Exception as e:
                print(f"  Lỗi khi xóa {img_path}: {e}")
    
    print(f"\nTổng cộng đã xóa: {total_removed} ảnh")
    return total_removed

def main():
    """
    Hàm chính để chạy chương trình
    """
    # Cấu hình - Test với folder IMG_5241 trước
    image_folder = 'train_images/IMG_5241'  # Folder chứa ảnh
    threshold = 5  # Ngưỡng khoảng cách Hamming
    backup_folder = 'backup_images'  # Folder backup (tùy chọn)
    
    print("=== CHƯƠNG TRÌNH TÌM VÀ XÓA ẢNH TRÙNG LẶP ===")
    print(f"Folder ảnh: {image_folder}")
    print(f"Ngưỡng similarity: {threshold}")
    print(f"Folder backup: {backup_folder}")
    print("-" * 50)
    
    # Kiểm tra folder tồn tại
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} không tồn tại!")
        return
    
    # Tìm các ảnh giống nhau
    similar_groups = find_similar_images(image_folder, threshold)
    
    if not similar_groups:
        print("Không tìm thấy ảnh trùng lặp nào!")
        return
    
    print(f"\nTìm thấy {len(similar_groups)} nhóm ảnh trùng lặp:")
    total_duplicates = sum(len(group) - 1 for group in similar_groups)
    print(f"Tổng cộng có {total_duplicates} ảnh trùng lặp có thể xóa")
    
    # Hỏi người dùng có muốn xóa không
    response = input("\nBạn có muốn xóa các ảnh trùng lặp không? (y/n): ")
    if response.lower() in ['y', 'yes', 'có']:
        removed_count = remove_duplicate_images(
            similar_groups, 
            keep_first=True, 
            backup_folder=backup_folder
        )
        print(f"\nHoàn thành! Đã xóa {removed_count} ảnh trùng lặp.")
    else:
        print("Đã hủy thao tác xóa.")

if __name__ == "__main__":
    main()

