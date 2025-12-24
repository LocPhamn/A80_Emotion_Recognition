"""
Script để kiểm tra và cài đặt dependencies cho demo_2.py
"""

import subprocess
import sys

def check_package(package_name):
    """Kiểm tra xem package đã được cài đặt chưa"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Cài đặt package"""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def main():
    print("=" * 60)
    print("Checking dependencies for demo_2.py (YOLOv11 + ByteTrack)")
    print("=" * 60)
    
    # Danh sách các package cần thiết
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'ultralytics': 'ultralytics',
        'loguru': 'loguru',
        'lap': 'lap',
        'cython_bbox': 'cython_bbox',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'scipy': 'scipy',
        'filterpy': 'filterpy'
    }
    
    missing_packages = []
    
    print("\n1. Checking packages...")
    print("-" * 60)
    
    for import_name, install_name in required_packages.items():
        status = "✓ INSTALLED" if check_package(import_name) else "✗ MISSING"
        print(f"{install_name:20} : {status}")
        
        if not check_package(import_name):
            missing_packages.append(install_name)
    
    if missing_packages:
        print("\n2. Missing packages found:")
        print("-" * 60)
        for pkg in missing_packages:
            print(f"  - {pkg}")
        
        response = input("\nDo you want to install missing packages? (y/n): ")
        
        if response.lower() == 'y':
            print("\n3. Installing packages...")
            print("-" * 60)
            for pkg in missing_packages:
                try:
                    install_package(pkg)
                    print(f"✓ {pkg} installed successfully")
                except Exception as e:
                    print(f"✗ Error installing {pkg}: {e}")
        else:
            print("\nSkipping installation. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\n✓ All required packages are installed!")
    
    print("\n4. Setting up ByteTrack...")
    print("-" * 60)
    
    # Kiểm tra xem đã setup ByteTrack chưa
    try:
        import yolox
        print("✓ ByteTrack (yolox) is already installed")
    except ImportError:
        print("✗ ByteTrack (yolox) not found")
        print("\nPlease run the following commands:")
        print("  cd ByteTrack")
        print("  pip install -e .")
        print("\nOr run this script from the correct directory:")
        
        import os
        bytetrack_path = os.path.join(os.path.dirname(__file__), 'ByteTrack')
        if os.path.exists(bytetrack_path):
            response = input("\nByteTrack folder found. Install now? (y/n): ")
            if response.lower() == 'y':
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", bytetrack_path])
                    print("✓ ByteTrack installed successfully")
                except Exception as e:
                    print(f"✗ Error installing ByteTrack: {e}")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python demo_2.py")
    print("\nFor more options:")
    print("  python demo_2.py --help")

if __name__ == "__main__":
    main()
