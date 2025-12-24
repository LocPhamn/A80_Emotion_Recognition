"""
Script để fix lỗi np.float deprecated trong ByteTrack
Thay thế tất cả np.float thành np.float64 trong các file Python
"""

import os
import re

def fix_np_float_in_file(filepath):
    """Fix np.float deprecated trong một file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Đếm số lượng thay thế
        original_content = content
        
        # Thay thế np.float (không phải np.float64, np.float32, etc.)
        # Pattern: np.float theo sau bởi không phải chữ số
        content = re.sub(r'np\.float(?![0-9])', 'np.float64', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            count = len(re.findall(r'np\.float(?![0-9])', original_content))
            print(f"✓ Fixed {count} occurrences in: {filepath}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        return False

def main():
    print("=" * 70)
    print("Fixing np.float deprecated warnings in ByteTrack")
    print("=" * 70)
    
    # Tìm thư mục ByteTrack
    bytetrack_dir = os.path.join(os.path.dirname(__file__), 'ByteTrack')
    
    if not os.path.exists(bytetrack_dir):
        print(f"Error: ByteTrack directory not found at {bytetrack_dir}")
        return
    
    print(f"\nScanning: {bytetrack_dir}")
    print("-" * 70)
    
    fixed_files = []
    scanned_files = 0
    
    # Duyệt qua tất cả file .py trong ByteTrack
    for root, dirs, files in os.walk(bytetrack_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                scanned_files += 1
                
                if fix_np_float_in_file(filepath):
                    fixed_files.append(filepath)
    
    print("-" * 70)
    print(f"\nSummary:")
    print(f"  Scanned: {scanned_files} Python files")
    print(f"  Fixed: {len(fixed_files)} files")
    
    if fixed_files:
        print(f"\nFixed files:")
        for f in fixed_files:
            rel_path = os.path.relpath(f, os.path.dirname(__file__))
            print(f"  - {rel_path}")
    else:
        print("\n✓ No files needed fixing (all already up to date)")
    
    print("\n" + "=" * 70)
    print("Done! You can now run demo_2.py without np.float warnings.")
    print("=" * 70)

if __name__ == "__main__":
    main()
