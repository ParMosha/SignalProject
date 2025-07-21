import cv2
import numpy as np
import time
from ultralytics import YOLO

class YOLOTracker:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.last_detection_time = 0
        self.current_object = None
        self.search_region = None
        self.tracked_class = None
        self.detection_interval = 0.01
        self.search_region_expansion = 1.5
        self.confidence_threshold = 0.5

    def detect_objects(self, frame, search_region=None):
        if search_region is not None:
            x, y, w, h = search_region
            frame = frame[y:y+h, x:x+w]
        
        results = self.model(frame, verbose=False)
        objects = []
        
        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()
                if confidence > self.confidence_threshold:
                    class_id = int(box.cls.item())
                    if self.tracked_class is None or class_id == self.tracked_class:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        objects.append({
                            'class_id': class_id,
                            'class_name': self.model.names[class_id],
                            'confidence': confidence,
                            'box': (x1, y1, x2-x1, y2-y1)
                        })
        
        if search_region is not None:
            for obj in objects:
                x1, y1, w, h = obj['box']
                obj['box'] = (x1 + search_region[0], y1 + search_region[1], w, h)
        
        return objects

    def update(self, frame):
        current_time = time.time()
        
        if (current_time - self.last_detection_time >= self.detection_interval or 
            self.current_object is None):
            
            search_region = None
            if self.current_object is not None:
                x, y, w, h = self.current_object['box']
                center_x, center_y = x + w//2, y + h//2
                new_size = int(max(w, h) * self.search_region_expansion)
                x1 = max(0, center_x - new_size//2)
                y1 = max(0, center_y - new_size//2)
                x2 = min(frame.shape[1], center_x + new_size//2)
                y2 = min(frame.shape[0], center_y + new_size//2)
                search_region = (x1, y1, x2-x1, y2-y1)
            
            objects = self.detect_objects(frame, search_region)
            self.last_detection_time = current_time
            
            if objects:
                self.current_object = max(objects, key=lambda x: x['confidence'])
                self.tracked_class = self.current_object['class_id']
                return True, self.current_object
        
        return False, self.current_object

def select_video_source():
    print("1. استفاده از دوربین لپتاپ")
    print("2. استفاده از فایل ویدیویی")
    while True:
        choice = input("لطفاً گزینه مورد نظر را انتخاب کنید (1 یا 2): ")
        if choice == '1':
            return 0
        elif choice == '2':
            video_path = input("لطفاً مسیر کامل فایل ویدیویی را وارد کنید: ").strip('"\'')
            return video_path
        else:
            print("گزینه نامعتبر! لطفاً 1 یا 2 وارد کنید")

def main():
    video_source = select_video_source()
    tracker = YOLOTracker('yolov8n.pt')
    
    if isinstance(video_source, str):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("خطا: نمی‌توان منبع ویدیو را باز کرد")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("خطا: نمی‌توان فریم را خواند")
        cap.release()
        return
    
    bbox = cv2.selectROI("انتخاب شیء برای ردیابی", frame, False)
    cv2.destroyWindow("انتخاب شیء برای ردیابی")
    tracker.initialize(frame, bbox)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        detected, obj = tracker.update(frame)
        
        if obj is not None:
            x, y, w, h = obj['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{obj['class_name']}: {obj['confidence']:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            if tracker.search_region is not None:
                x_sr, y_sr, w_sr, h_sr = tracker.search_region
                cv2.rectangle(frame, (x_sr, y_sr), (x_sr+w_sr, y_sr+h_sr), (255, 0, 0), 1)
        
        status = "تشخیص" if detected else "ردیابی"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        fps = 1 / (time.time() - tracker.last_detection_time) if tracker.last_detection_time else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("سیستم ردیابی YOLO", frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()