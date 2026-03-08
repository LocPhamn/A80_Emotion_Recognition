"""
Demo webcam LOCAL - không cần API/network
Chạy face tracking + emotion detection trực tiếp từ webcam
Hữu ích khi demo mà mạng không ổn định

Phím điều khiển:
    - R: Bắt đầu/Dừng recording
    - Q: Thoát
    - S: Chụp screenshot
    - Space: Pause/Resume
"""

import os
import sys
import cv2
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import tracker
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ai.datn.script.datn_ai import FaceEmotionTracker

# ANSI Color codes cho console output
class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Emotion-specific colors
    ANGRY = '\033[91m'      # Red
    DISGUST = '\033[95m'    # Magenta
    FEAR = '\033[35m'       # Purple
    HAPPY = '\033[92m'      # Green
    NEUTRAL = '\033[37m'    # Light gray
    SAD = '\033[94m'        # Blue
    SURPRISE = '\033[93m'   # Yellow

# Import config
try:
    from demo_config import (
        WEBCAM_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG, 
        DISPLAY_CONFIG, EMOTION_LABELS
    )
    WEBCAM_ID = WEBCAM_CONFIG["webcam_id"]
    DISPLAY_WIDTH = WEBCAM_CONFIG["display_width"]
    DISPLAY_HEIGHT = WEBCAM_CONFIG["display_height"]
    RECORDING_FPS = WEBCAM_CONFIG["recording_fps"]
    OUTPUT_DIR = Path(OUTPUT_CONFIG["output_dir"])
    EMOTION_DISPLAY_LABELS = EMOTION_LABELS["display_labels"]
    print(f"{Colors.GREEN}✅ Loaded config from demo_config.py{Colors.RESET}")
except ImportError:
    # Fallback to default values
    WEBCAM_ID = 0
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    RECORDING_FPS = 15.0
    OUTPUT_DIR = Path("./demo_recordings")
    EMOTION_DISPLAY_LABELS = {
        'angry': 'Tức giận',
        'disgust': 'Chán nản',
        'fear': 'Sợ hãi',
        'happy': 'Hạnh phúc',
        'neutral': 'Trung tính',
        'sad': 'Buồn bã',
        'surprise': 'Bất ngờ'
    }
    print(f"{Colors.YELLOW}⚠️ Config file not found, using default values{Colors.RESET}")

# Directories
OUTPUT_DIR.mkdir(exist_ok=True)


class WebcamDemo:
    """Demo webcam với face tracking và emotion detection"""
    
    def __init__(self, webcam_id=0):
        print("=" * 60)
        print(f"{Colors.BOLD}{Colors.CYAN}🎥 WEBCAM DEMO - Face Tracking + Emotion Detection{Colors.RESET}")
        print("=" * 60)
        print(f"\n{Colors.YELLOW}📋 Phím điều khiển:{Colors.RESET}")
        print(f"   {Colors.GREEN}R:{Colors.RESET} Bắt đầu/Dừng recording")
        print(f"   {Colors.GREEN}S:{Colors.RESET} Chụp screenshot")
        print(f"   {Colors.GREEN}Space:{Colors.RESET} Pause/Resume")
        print(f"   {Colors.GREEN}Q:{Colors.RESET} Thoát\n")
        
        # Khởi tạo tracker
        print(f"{Colors.YELLOW}🔧 Đang khởi tạo FaceEmotionTracker...{Colors.RESET}")
        self.tracker = FaceEmotionTracker()
        
        # Mở webcam
        print(f"{Colors.YELLOW}📹 Đang mở webcam {webcam_id}...{Colors.RESET}")
        self.cap = cv2.VideoCapture(webcam_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"{Colors.RED}❌ Không thể mở webcam {webcam_id}{Colors.RESET}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        
        # Get actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"{Colors.GREEN}✅ Webcam đã sẵn sàng: {self.width}x{self.height}{Colors.RESET}")
        
        # Recording state
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.recording_frame_count = 0
        self.session_id = None
        
        # Statistics tracking (giống AI server)
        self.visited_ids = set()
        self.track_final_emotion = {}
        
        # Pause state
        self.paused = False
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        
    def start_recording(self):
        """Bắt đầu recording"""
        if self.recording:
            print(f"{Colors.YELLOW}⚠️ Đã đang recording!{Colors.RESET}")
            return
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"demo_{self.session_id}.mp4"
        video_path = OUTPUT_DIR / video_filename
        
        # Tạo video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            RECORDING_FPS,
            (self.width, self.height)
        )
        
        if not self.video_writer.isOpened():
            print(f"{Colors.RED}❌ Không thể tạo video writer!{Colors.RESET}")
            self.video_writer = None
            return
        
        self.recording = True
        self.recording_start_time = time.time()
        self.recording_frame_count = 0
        
        # Reset statistics cho session mới
        self.visited_ids.clear()
        self.track_final_emotion.clear()
        
        print(f"{Colors.RED}{Colors.BOLD}🔴 RECORDING STARTED: {video_filename}{Colors.RESET}")
        
    def stop_recording(self):
        """Dừng recording và hiển thị thống kê"""
        if not self.recording:
            print(f"{Colors.YELLOW}⚠️ Không có recording nào đang chạy!{Colors.RESET}")
            return
        
        self.recording = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        duration = time.time() - self.recording_start_time
        
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}⏹️ RECORDING STOPPED{Colors.RESET}")
        print(f"   Session ID: {Colors.CYAN}{self.session_id}{Colors.RESET}")
        print(f"   Duration: {Colors.GREEN}{duration:.1f}s{Colors.RESET}")
        print(f"   Frames: {Colors.GREEN}{self.recording_frame_count}{Colors.RESET}")
        print(f"   FPS: {Colors.GREEN}{self.recording_frame_count / duration:.1f}{Colors.RESET}")
        
        # Tính statistics (giống logic trong datn_ai.py)
        self._calculate_and_display_statistics()
        
    def _calculate_and_display_statistics(self):
        """Tính và hiển thị thống kê cảm xúc (giống AI server)"""
        total_visitor = len(self.visited_ids)
        
        if total_visitor == 0:
            print("   ℹ️ Không có người nào được phát hiện")
            return
        
        # Đếm emotion
        emotion_visitors = defaultdict(int)
        for track_id in self.visited_ids:
            if track_id in self.track_final_emotion:
                emotion = self.track_final_emotion[track_id]
                emotion_visitors[emotion] += 1
        
        # Mapping màu sắc cho từng emotion
        emotion_colors = {
            'angry': Colors.ANGRY,
            'disgust': Colors.DISGUST,
            'fear': Colors.FEAR,
            'happy': Colors.HAPPY,
            'neutral': Colors.NEUTRAL,
            'sad': Colors.SAD,
            'surprise': Colors.SURPRISE
        }
        
        # Tính phần trăm
        print(f"\n{Colors.BOLD}{Colors.CYAN}📊 THỐNG KÊ:{Colors.RESET}")
        print(f"{Colors.BOLD}   Tổng số người: {Colors.GREEN}{total_visitor}{Colors.RESET}")
        print(f"\n   Tỉ lệ cảm xúc (%):")
        
        total_percentage = 0.0
        for emotion_en in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
            count = emotion_visitors.get(emotion_en, 0)
            ratio = (count / total_visitor * 100) if total_visitor > 0 else 0.0
            emotion_vi = EMOTION_DISPLAY_LABELS.get(emotion_en, emotion_en)
            color = emotion_colors.get(emotion_en, Colors.WHITE)
            
            # In ra với màu tương ứng
            print(f"   - {color}{emotion_vi:15s}: {ratio:6.2f}% ({count} người){Colors.RESET}")
            total_percentage += ratio
        
        print(f"   {Colors.BOLD}{Colors.GREEN}✓ Tổng: {total_percentage:.2f}%{Colors.RESET}\n")
        
    def take_screenshot(self, frame):
        """Chụp screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = OUTPUT_DIR / f"screenshot_{timestamp}.jpg"
        cv2.imwrite(str(screenshot_path), frame)
        print(f"{Colors.CYAN}📸 Screenshot saved: {screenshot_path.name}{Colors.RESET}")
        
    def run(self):
        """Main loop"""
        print(f"\n{Colors.GREEN}{Colors.BOLD}▶️ Demo đang chạy... (Nhấn Q để thoát){Colors.RESET}\n")
        
        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        print(f"{Colors.RED}❌ Không thể đọc frame từ webcam!{Colors.RESET}")
                        break
                    
                    # Xử lý frame với AI
                    result = self.tracker.process_frame(frame)
                    processed_frame = result['frame']
                    
                    # Cập nhật statistics (tracking unique visitors + emotion)
                    for track in result['tracks']:
                        track_id = track['id']
                        emotion_vi = track['emotion']
                        
                        # Thêm vào visited IDs
                        self.visited_ids.add(track_id)
                        
                        # Convert emotion sang tiếng Anh
                        emotion_en = self.tracker.emotion_to_english.get(emotion_vi, 'neutral')
                        
                        # Lưu emotion cuối cùng (ghi đè)
                        self.track_final_emotion[track_id] = emotion_en
                    
                    # Ghi vào video nếu đang recording
                    if self.recording and self.video_writer:
                        self.video_writer.write(processed_frame)
                        self.recording_frame_count += 1
                    
                    # Vẽ recording indicator
                    if self.recording:
                        # Vẽ chấm đỏ nhấp nháy
                        if int(time.time() * 2) % 2:
                            cv2.circle(processed_frame, (30, 30), 15, (0, 0, 255), -1)
                        cv2.putText(processed_frame, "REC", (55, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Hiển thị thời gian recording
                        rec_duration = time.time() - self.recording_start_time
                        rec_time_text = f"{int(rec_duration // 60):02d}:{int(rec_duration % 60):02d}"
                        cv2.putText(processed_frame, rec_time_text, (55, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Hiển thị số người đã tracking
                        visitor_text = f"Visitors: {len(self.visited_ids)}"
                        cv2.putText(processed_frame, visitor_text, (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Hiển thị hướng dẫn
                    help_text = "R:Record | S:Screenshot | Space:Pause | Q:Quit"
                    cv2.putText(processed_frame, help_text, (10, self.height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Hiển thị frame
                    cv2.imshow('Webcam Demo - Face Emotion Tracking', processed_frame)
                    
                    self.frame_count += 1
                
                else:
                    # Paused - vẽ text thông báo
                    paused_frame = frame.copy()
                    cv2.putText(paused_frame, "PAUSED", (self.width // 2 - 100, self.height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                    cv2.imshow('Webcam Demo - Face Emotion Tracking', paused_frame)
                
                # Xử lý phím bấm
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print(f"\n{Colors.YELLOW}👋 Đang thoát...{Colors.RESET}")
                    break
                    
                elif key == ord('r') or key == ord('R'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                        
                elif key == ord('s') or key == ord('S'):
                    if not self.paused:
                        self.take_screenshot(processed_frame)
                        
                elif key == 32:  # Space bar
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    color = Colors.YELLOW if self.paused else Colors.GREEN
                    print(f"{color}⏸️ {status}{Colors.RESET}")
        
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️ Interrupted by user (Ctrl+C){Colors.RESET}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print(f"\n{Colors.CYAN}🧹 Cleaning up...{Colors.RESET}")
        
        # Stop recording nếu còn đang chạy
        if self.recording:
            self.stop_recording()
        
        # Release webcam
        if self.cap:
            self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Statistics tổng quát
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}📊 Session Summary:{Colors.RESET}")
        print(f"   Total frames processed: {Colors.GREEN}{self.frame_count}{Colors.RESET}")
        print(f"   Total time: {Colors.GREEN}{elapsed_time:.1f}s{Colors.RESET}")
        print(f"   Average FPS: {Colors.GREEN}{avg_fps:.1f}{Colors.RESET}")
        print(f"   Recordings saved to: {Colors.YELLOW}{OUTPUT_DIR}{Colors.RESET}")
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Demo ended. Goodbye!{Colors.RESET}\n")


def main():
    """Entry point"""
    try:
        demo = WebcamDemo(webcam_id=WEBCAM_ID)
        demo.run()
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
