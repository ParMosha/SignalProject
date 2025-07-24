import cv2
import time
from ultralytics import YOLO

class YOLOMultiTracker:
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        confidence_threshold: float = 0.8,
        detection_interval: float = 0.1
    ):
        self.model = YOLO(model_path)
        self.conf_thr = confidence_threshold
        self.det_interval = detection_interval
        self.last_det_time = 0.0
        self.current_objects = []

    def detect_objects(self, frame):
        """
        Detect all objects in the full frame above confidence threshold.
        """
        results = self.model(frame, verbose=False)
        objs = []

        for result in results:
            for box in result.boxes:
                conf = box.conf.item()
                if conf < self.conf_thr:
                    continue

                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                objs.append({
                    'class_id':   cls_id,
                    'class_name': self.model.names[cls_id],
                    'confidence': conf,
                    'box':        (x1, y1, x2 - x1, y2 - y1)
                })

        return objs

    def update(self, frame):
        """
        If enough time has passed, rerun detection on full frame.
        Returns (did_detect: bool, list_of_objects: list).
        """
        now = time.time()
        if now - self.last_det_time >= self.det_interval:
            self.current_objects = self.detect_objects(frame)
            self.last_det_time = now
            return True, self.current_objects

        return False, self.current_objects


def main():
  
    print("Select source:")
    print("  1) Camera")
    print("  2) Video file")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        source = input("Enter path to video file: ").strip()
    else:
        source = 0  

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open source: {source}")
        return

    tracker = YOLOMultiTracker(
        model_path='yolov8n.pt',
        confidence_threshold=0.5,
        detection_interval=0.1
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot read frame.")
            break

        detected, objects = tracker.update(frame)

        for obj in objects:
            x, y, w, h = obj['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{obj['class_name']}: {obj['confidence']:.2f}"
            cv2.putText(frame, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        status = "Detecting" if detected else "Tracking"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        fps = 1.0 / max(1e-6, time.time() - tracker.last_det_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("YOLO Multi-Object Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
