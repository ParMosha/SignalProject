import cv2
import time
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import threading
import queue
from collections import deque

class VideoProcessor:
    def __init__(self, source, model_path='yolov8n.pt', confidence_threshold=0.5):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        self.fps = max(1, self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.model = YOLO(model_path)
        self.conf_thr = confidence_threshold
        
        # Buffering system
        self.processed_frames = deque(maxlen=int(self.fps * 3))  # 3 second buffer
        self.processing_queue = queue.Queue(maxsize=int(self.fps * 2))  # Processing queue
        
        self.running = False
        self.current_frame_pos = 0
        self.tracked_objects = {}
        self.last_detection_time = 0
        self.detection_interval = 0.2  # Detection every 0.2 seconds
        
    def preprocess_video(self):
        """Initial video analysis to detect objects"""
        print("Starting video preprocessing...")
        
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process first frame
        ret, frame = self.cap.read()
        if ret:
            initial_objects = self.detect_objects(frame, full_frame=True)
            self.tracked_objects = {obj['track_id']: obj for obj in initial_objects}
        
        # Process remaining frames at intervals
        frame_step = max(1, int(self.fps * self.detection_interval))
        current_frame = 0
        
        while self.running and current_frame < self.frame_count:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if current_frame % frame_step == 0:
                self.process_frame(frame, is_preprocessing=True)
            
            current_frame += 1
        
        # Reset to beginning for playback
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_pos = 0
        print("Preprocessing completed.")
    
    def detect_objects(self, frame, full_frame=False):
        """Detect objects in frame using YOLO"""
        if full_frame:
            results = self.model.track(frame, persist=True, verbose=False)
        else:
            regions = self.get_detection_regions()
            results = []
            for region in regions:
                x, y, w, h = region
                roi = frame[y:y+h, x:x+w]
                roi_results = self.model.track(roi, persist=True, verbose=False)
                for res in roi_results:
                    for box in res.boxes:
                        # Convert ROI coordinates to main frame
                        box.xyxy[0][0] += x
                        box.xyxy[0][1] += y
                        box.xyxy[0][2] += x
                        box.xyxy[0][3] += y
                results.extend(roi_results)
        
        objs = []
        for res in results:
            for box in res.boxes:
                conf = box.conf.item()
                if conf < self.conf_thr:
                    continue
                cls_id = int(box.cls.item())
                track_id = int(box.id.item()) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                objs.append({
                    'class_id': cls_id,
                    'class_name': self.model.names[cls_id],
                    'confidence': conf,
                    'box': (x1, y1, x2-x1, y2-y1),
                    'track_id': track_id,
                    'last_seen': time.time()
                })
        return objs
    
def get_detection_regions(self):
    """Get regions of interest for detection"""
    regions = []
    
    # Regions around tracked objects
    for obj in self.tracked_objects.values():
        x, y, w, h = obj['box']
        expanded_region = (
            max(0, x - w//2), 
            max(0, y - h//2), 
            min(self.width, w*2), 
            min(self.height, h*2)
        )
        regions.append(expanded_region)
    
    # Add margin regions (20% of frame)
    margin = 0.2
    margin_w = int(self.width * margin)
    margin_h = int(self.height * margin)
    
    regions.extend([
        (0, 0, self.width, margin_h),  # Top
        (0, self.height-margin_h, self.width, margin_h),  # Bottom
        (0, 0, margin_w, self.height),  # Left
        (self.width-margin_w, 0, margin_w, self.height)   # Right
    ])  # اینجا براکت بسته شد
    
    return regions
    
    def process_frame(self, frame, is_preprocessing=False):
        """Process a frame and update tracking"""
        current_time = time.time()
        
        # Detect objects at intervals
        if current_time - self.last_detection_time >= self.detection_interval or is_preprocessing:
            new_objects = self.detect_objects(frame, full_frame=is_preprocessing)
            self.last_detection_time = current_time
            
            # Update tracked objects
            updated_objects = {}
            for new_obj in new_objects:
                track_id = new_obj['track_id']
                if track_id in self.tracked_objects:
                    # Update existing object
                    updated_objects[track_id] = {**self.tracked_objects[track_id], **new_obj}
                else:
                    # Add new object
                    updated_objects[track_id] = new_obj
            
            # Remove stale objects
            self.tracked_objects = {
                k: v for k, v in updated_objects.items() 
                if (current_time - v['last_seen']) < 2.0  # 2 seconds timeout
            }
        
        # Draw bounding boxes
        output_frame = frame.copy()
        for obj in self.tracked_objects.values():
            x, y, w, h = obj['box']
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{obj['class_name']} ID:{obj['track_id']} ({obj['confidence']:.2f})"
            cv2.putText(output_frame, label, (x, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_frame, list(self.tracked_objects.values())
    
    def processing_worker(self):
        """Background processing thread"""
        while self.running:
            try:
                frame_info = self.processing_queue.get(timeout=0.5)
                frame, frame_pos = frame_info
                
                processed_frame, objects = self.process_frame(frame)
                
                self.processed_frames.append({
                    'frame': processed_frame,
                    'objects': objects,
                    'position': frame_pos
                })
                
                self.processing_queue.task_done()
            except queue.Empty:
                continue
    
    def start_processing(self):
        """Start video processing pipeline"""
        self.running = True
        
        # Start preprocessing
        self.preprocess_video()
        
        # Start processing thread
        self.worker_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.worker_thread.start()
        
        # Feed frames to processing queue
        while self.running and self.current_frame_pos < self.frame_count:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            try:
                self.processing_queue.put((frame, self.current_frame_pos), timeout=0.1)
                self.current_frame_pos += 1
            except queue.Full:
                time.sleep(0.01)
    
    def stop_processing(self):
        """Cleanup resources"""
        self.running = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join()
        self.cap.release()
    
    def get_frame(self, position):
        """Get processed frame by position"""
        for frame_data in self.processed_frames:
            if frame_data['position'] == position:
                return frame_data['frame'], frame_data['objects']
        return None, None

def select_file():
    """Open file dialog"""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

def main():
    print("Select source:")
    print("1) Camera")
    print("2) Video file (manual path)")
    print("3) Browse video file")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == '1':
        source = 0
    elif choice == '2':
        source = input("Enter video file path: ").strip('"')
    elif choice == '3':
        source = select_file()
        if not source:
            print("No file selected!")
            return
    else:
        print("Invalid choice!")
        return

    try:
        processor = VideoProcessor(source)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Start processing
    processor.start_processing()

    # Playback control
    current_pos = 0
    paused = False
    last_frame_time = time.time()
    
    while True:
        if not paused:
            # Get processed frame
            frame, objects = processor.get_frame(current_pos)
            
            if frame is not None:
                # Display frame
                cv2.imshow("Object Tracker", frame)
                
                # Maintain correct playback speed
                current_time = time.time()
                elapsed = current_time - last_frame_time
                target_delay = 1.0 / processor.fps
                
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
                
                last_frame_time = time.time()
                current_pos += 1
            else:
                # Wait for frames to be processed
                time.sleep(0.01)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord('p'):  # Pause/resume
            paused = not paused
            if not paused:
                last_frame_time = time.time()  # Reset timer after pause
        elif key == ord('r'):  # Rewind
            current_pos = max(0, current_pos - int(processor.fps * 2))  # 2 seconds back
        elif key == ord('f'):  # Fast forward
            current_pos = min(processor.frame_count-1, current_pos + int(processor.fps * 2))  # 2 seconds forward
    
    # Cleanup
    processor.stop_processing()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()