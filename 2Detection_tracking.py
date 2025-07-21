import cv2
import numpy as np
import time
from ultralytics import YOLO

class YOLOTracker:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.last_detection_time = 0
        self.current_object = None
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
                conf = box.conf.item()
                if conf < self.confidence_threshold:
                    continue

                cls_id = int(box.cls.item())
                # Only keep our tracked class if already set
                if self.tracked_class is not None and cls_id != self.tracked_class:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                objects.append({
                    'class_id': cls_id,
                    'class_name': self.model.names[cls_id],
                    'confidence': conf,
                    'box': (x1, y1, x2 - x1, y2 - y1)
                })

        # Shift coordinates back if we cropped for a search region
        if search_region is not None:
            sx, sy, _, _ = search_region
            for obj in objects:
                x1, y1, w, h = obj['box']
                obj['box'] = (x1 + sx, y1 + sy, w, h)

        return objects

    def update(self, frame):
        now = time.time()
        if (now - self.last_detection_time >= self.detection_interval
                or self.current_object is None):
            search_region = None
            if self.current_object:
                x, y, w, h = self.current_object['box']
                cx, cy = x + w // 2, y + h // 2
                new_size = int(max(w, h) * self.search_region_expansion)
                x1 = max(0, cx - new_size // 2)
                y1 = max(0, cy - new_size // 2)
                x2 = min(frame.shape[1], cx + new_size // 2)
                y2 = min(frame.shape[0], cy + new_size // 2)
                search_region = (x1, y1, x2 - x1, y2 - y1)

            objs = self.detect_objects(frame, search_region)
            self.last_detection_time = now

            if objs:
                # pick highest-confidence
                self.current_object = max(objs, key=lambda o: o['confidence'])
                self.tracked_class = self.current_object['class_id']
                return True, self.current_object

        return False, self.current_object

def main():
    # --- choose input source ---
    print("Select source:")
    print("1) Camera")
    print("2) Video file")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        path = input("Enter path to video file: ").strip()
        source = path
    else:
        # default to webcam 0
        source = 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open source {source}")
        return

    tracker = YOLOTracker(model_path='yolov8n.pt')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or cannot read frame.")
            break

        detected, obj = tracker.update(frame)

        if obj:
            x, y, w, h = obj['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{obj['class_name']} {obj['confidence']:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        status = "Detecting" if detected else "Tracking"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # compute FPS based on last detection interval
        fps = 1.0 / max(1e-6, (time.time() - tracker.last_detection_time))
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("YOLO Object Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
