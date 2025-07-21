import cv2
import time
import threading
import collections
from ultralytics import YOLO

class YOLOMultiTracker:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, detection_interval=0.1):
        self.model = YOLO(model_path)
        self.conf_thr = confidence_threshold
        self.det_interval = detection_interval
        self.last_det_time = 0.0
        self.current_objects = []

    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)
        objs = []
        for res in results:
            for box in res.boxes:
                conf = box.conf.item()
                if conf < self.conf_thr:
                    continue
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                objs.append({
                    'class_id':   cls_id,
                    'class_name': self.model.names[cls_id],
                    'confidence': conf,
                    'box':        (x1, y1, x2-x1, y2-y1)
                })
        return objs

    def update(self, frame):
        now = time.time()
        if now - self.last_det_time >= self.det_interval:
            self.current_objects = self.detect_objects(frame)
            self.last_det_time = now
            return True, self.current_objects
        return False, self.current_objects

class ChunkWorker(threading.Thread):
    """
    Thread برای پیش‌پردازش یک chunk یک‌ثانیه‌ای
    نتایج در result_list ذخیره می‌شود.
    """
    def __init__(self, frames, result_list):
        super().__init__(daemon=True)
        self.frames = frames
        self.results = result_list
        self.tracker = YOLOMultiTracker(
            model_path='yolov8n.pt',
            confidence_threshold=0.5,
            detection_interval=0.1
        )

    def run(self):
        for i, frame in enumerate(self.frames):
            did, objs = self.tracker.update(frame)
            self.results[i] = (did, objs)

def main():
    print("Select source:")
    print("  1) Camera")
    print("  2) Video file")
    choice = input("Enter 1 or 2: ").strip()
    source = 0 if choice == '1' else input("Enter video path: ").strip()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open {source}")
        return

    # فقط برای ویدیو
    if source != 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        chunk_size = int(fps * 1.0)              # فریم‌های معادل یک ثانیه
        frame_delay = int(1000 / fps)

        # خواندن دو chunk اولیه
        def read_chunk():
            buf = []
            for _ in range(chunk_size):
                ret, frm = cap.read()
                if not ret:
                    break
                buf.append(frm.copy())
            return buf

        frames0 = read_chunk()
        frames1 = read_chunk()
        if not frames0:
            return

        # رزرو نتایج
        results0 = [None] * len(frames0)
        results1 = [None] * len(frames1)

        # پردازش Chunk0 سینک (همان ترد اصلی)
        wk0 = ChunkWorker(frames0, results0)
        wk0.run()

        # استارت پردازش Chunk1 در ترد جدا
        wk1 = ChunkWorker(frames1, results1)
        wk1.start()

        current_frames, current_results = frames0, results0
        next_worker, next_frames, next_results = wk1, frames1, results1

        # حلقه نمایش و پیش‌پردازش بعدی
        while current_frames:
            # نمایش chunk فعلی
            for frm, res in zip(current_frames, current_results):
                did, objs = res
                for obj in objs:
                    x, y, w, h = obj['box']
                    cv2.rectangle(frm, (x, y), (x+w, y+h), (0,255,0), 2)
                    lbl = f"{obj['class_name']}:{obj['confidence']:.2f}"
                    cv2.putText(frm, lbl, (x, y-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.imshow("Tracker", frm)
                if cv2.waitKey(frame_delay) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # منتظر اتمام کار ترد بعدی می‌مانیم
            next_worker.join()

            # آماده‌سازی برای chunk بعدی
            current_frames, current_results = next_frames, next_results

            # خواندن و استارت پردازش chunk جدید
            new_frames = read_chunk()
            if not new_frames:
                break
            new_results = [None] * len(new_frames)
            new_worker = ChunkWorker(new_frames, new_results)
            new_worker.start()

            # جابجایی
            next_worker, next_frames, next_results = new_worker, new_frames, new_results

        cap.release()
        cv2.destroyAllWindows()

    # برای وب‌کم: همان الگوریتم اصلی بدون تغییر
    else:
        tracker = YOLOMultiTracker(
            model_path='yolov8n.pt',
            confidence_threshold=0.5,
            detection_interval=0.1
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            det, objs = tracker.update(frame)
            for obj in objs:
                x, y, w, h = obj['box']
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                lbl = f"{obj['class_name']}:{obj['confidence']:.2f}"
                cv2.putText(frame, lbl, (x,y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            status = "Detecting" if det else "Tracking"
            cv2.putText(frame, status, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
