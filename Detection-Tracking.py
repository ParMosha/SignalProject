import cv2
import numpy as np
import time

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0],
                                             [0,0,0,1,0,0,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],
                                            [0,1,0,0,0,1,0,0],
                                            [0,0,1,0,0,0,1,0],
                                            [0,0,0,1,0,0,0,1],
                                            [0,0,0,0,1,0,0,0],
                                            [0,0,0,0,0,1,0,0],
                                            [0,0,0,0,0,0,1,0],
                                            [0,0,0,0,0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        
    def init(self, bbox):
        """مقداردهی اولیه با جعبه مرزی اولیه"""
        x, y, w, h = bbox
        self.kf.statePost = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        
    def predict(self):
        """پیش‌بینی موقعیت بعدی"""
        prediction = self.kf.predict()
        return (prediction[0][0], prediction[1][0], prediction[2][0], prediction[3][0])
        
    def update(self, bbox):
        """به‌روزرسانی با اندازه‌گیری جدید"""
        x, y, w, h = bbox
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)

class AdvancedTracker:
    """سیستم ردیابی پیشرفته ترکیبی"""
    def __init__(self):
        self.tracker = None
        self.kalman = KalmanFilter()
        self.lost_count = 0
        self.max_lost = 15
        self.scale_factor = 1.0
        self.prev_feature = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def initialize(self, frame, bbox):
        """مقداردهی اولیه ردیاب"""
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.kalman.init(bbox)
        self.prev_feature = self.calc_hist_feature(frame, bbox)
        
    def calc_hist_feature(self, frame, bbox):
        """محاسبه ویژگی هیستوگرام برای تشخیص تغییر مقیاس"""
        x, y, w, h = [int(v) for v in bbox]
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return 0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        return np.sum(hist[50:200])
        
    def adjust_scale(self, frame, bbox):
        """تنظیم خودکار اندازه جعبه مرزی"""
        current_feature = self.calc_hist_feature(frame, bbox)
        
        if self.prev_feature is not None and self.prev_feature > 0:
            scale_change = current_feature / self.prev_feature
            self.scale_factor *= scale_change
            self.scale_factor = np.clip(self.scale_factor, 0.5, 2.0)
            
        self.prev_feature = current_feature
        x, y, w, h = bbox
        return (x, y, w*self.scale_factor, h*self.scale_factor)
        
    def update(self, frame):
        """به‌روزرسانی موقعیت شیء"""
        # محاسبه FPS
        self.frame_count += 1
        if self.frame_count >= 10:
            self.fps = self.frame_count / (time.time() - self.start_time)
            self.start_time = time.time()
            self.frame_count = 0
            
        # ردیابی با CSRT
        success, bbox = self.tracker.update(frame)
        
        if success:
            self.lost_count = 0
            bbox = self.adjust_scale(frame, bbox)
            self.kalman.update(bbox)
            return True, bbox
        else:
            self.lost_count += 1
            if self.lost_count > self.max_lost:
                return False, None
            predicted = self.kalman.predict()
            return False, predicted

def select_video_source():
    """انتخاب منبع ویدیو"""
    print("1. Use laptop camera")
    print("2. Use video file")
    while True:
        choice = input("Please select an option (1 or 2): ")
        if choice == '1':
            return 0
        elif choice == '2':
            while True:
                video_path = input("Please enter video file path: ").strip('"\'')
                video_path = video_path.replace("\\", "/")  # اصلاح مسیر برای ویندوز
                if os.path.exists(video_path):
                    return video_path
                print(f"Error: File not found at {video_path}")

def main():
    """تابع اصلی اجرای سیستم ردیابی"""
    # انتخاب منبع ویدیو
    video_source = select_video_source()
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
        
    # خواندن اولین فریم
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        cap.release()
        return
    
    # انتخاب شیء برای ردیابی
    bbox = cv2.selectROI("Select Object to Track", frame, False)
    cv2.destroyWindow("Select Object to Track")
    
    # مقداردهی اولیه ردیاب
    tracker = AdvancedTracker()
    tracker.initialize(frame, bbox)
    
    # حلقه اصلی پردازش
    while True:
        ret, frame = cap.read()
        if not ret:
            # بازپخش ویدیو اگر از فایل استفاده می‌شود
            if video_source != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
            
        # ردیابی شیء
        success, bbox = tracker.update(frame)
        
        # نمایش نتایج
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            status = "Tracking"
            color = (0, 255, 0)
        else:
            status = "Searching..."
            color = (0, 0, 255)
            
        # نمایش اطلاعات
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, f"FPS: {tracker.fps:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, "Press ESC to exit", (frame.shape[1]-200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        
        cv2.imshow("Advanced Object Tracker", frame)
        
        # خروج با کلید ESC
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    main()