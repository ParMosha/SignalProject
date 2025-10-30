import cv2
import torch
import numpy as np
import os
import time

DETECTION_INTERVAL = 10
MAX_MISSED_FRAMES = 30


class YourObjectDetector:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True).to(self.device)
            self.model.conf = confidence_threshold
            self.model.iou = 0.45
            self.model.classes = [0]
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            print("Please ensure PyTorch, torchvision, and git are installed.")
            print("Also, check your internet connection to download the model.")
            exit()

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            detections.append(((x1, y1, w, h), float(conf), int(cls)))
        return detections


class MyCustomTracker:
    def __init__(self, initial_bbox, tracker_id):
        self.tracker_id = tracker_id
        self.bbox = initial_bbox
        self.active = True
        self.missing_frames_count = 0
        self.kalman = None
        self.prev_points = None
        self.prev_gray = None
        self.bbox_history = []
        self._init_kalman(initial_bbox)

    def _init_kalman(self, initial_bbox):
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]], np.float32)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        
        x, y, w, h = initial_bbox
        self.kalman.statePost = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], np.float32)
        self.kalman.statePre = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], np.float32)
        self.bbox = tuple(self.kalman.statePost[0:4, 0])
        self.update_history(self.bbox)

    def _init_optical_flow_points(self, frame_gray):
        x, y, w, h = map(int, self.bbox)
        if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= frame_gray.shape[1] and y + h <= frame_gray.shape[0]:
            roi = frame_gray[y:y+h, x:x+w]
            self.prev_points = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            if self.prev_points is not None:
                self.prev_points[:, 0, 0] += x
                self.prev_points[:, 0, 1] += y
                self.prev_gray = frame_gray.copy()
        else:
            self.prev_points = None
            self.prev_gray = None

    def update_with_detection(self, new_bbox, frame_gray):
        x, y, w, h = new_bbox
        self.kalman.correct(np.array([[x], [y], [w], [h]], np.float32))
        self.bbox = tuple(self.kalman.statePost[0:4, 0])
        self.missing_frames_count = 0
        self.active = True
        self._init_optical_flow_points(frame_gray)
        self.update_history(self.bbox)

    def update(self, current_frame_gray):
        if not self.active:
            return

        prediction = self.kalman.predict()
        optical_flow_measurement = None

        if self.prev_points is not None and self.prev_gray is not None and len(self.prev_points) > 0:
            next_points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_frame_gray, self.prev_points, None, **dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)))

            if next_points is not None and status is not None:
                good_new = next_points[status == 1]
                good_old = self.prev_points[status == 1]

                if len(good_new) >= 4:
                    M, _ = cv2.estimateAffine2D(good_old, good_new)
                    if M is not None:
                        x, y, w, h = self.bbox
                        corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32).reshape(-1, 1, 2)
                        transformed_corners = cv2.transform(corners, M).reshape(-1, 2)
                        x_min, y_min = np.min(transformed_corners, axis=0)
                        x_max, y_max = np.max(transformed_corners, axis=0)
                        optical_flow_measurement = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                self.prev_gray = current_frame_gray.copy()
                self.prev_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
            else:
                self.prev_points = None
        else:
            self._init_optical_flow_points(current_frame_gray)

        if optical_flow_measurement is not None:
            x, y, w, h = optical_flow_measurement
            self.kalman.correct(np.array([[x], [y], [w], [h]], np.float32))
            self.missing_frames_count = 0
        else:
            self.missing_frames_count += 1

        self.bbox = tuple(self.kalman.statePost[0:4, 0])
        self.update_history(self.bbox)

        if self.missing_frames_count > MAX_MISSED_FRAMES:
            self.active = False
            
    def update_history(self, bbox):
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > 5:
            self.bbox_history.pop(0)


def run_tracker(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    object_detector = YourObjectDetector()
    main_tracker = None
    frame_count = 0

    print("--- Starting object tracking. Press 'q' to exit. ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error fetching frame. Exiting.")
            break

        current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % DETECTION_INTERVAL == 0:
            print(f"--- Frame {frame_count}: Running full detection ---")
            raw_detections = object_detector.detect(frame)
            
            if raw_detections:
                main_detection = max(raw_detections, key=lambda det: det[0][2] * det[0][3])
                main_bbox, _, _ = main_detection
                print(f"  Main object detected with bbox {main_bbox}.")

                if main_tracker is None or not main_tracker.active:
                    print("  Creating a new main tracker.")
                    main_tracker = MyCustomTracker(main_bbox, tracker_id=1)
                else:
                    print("  Updating existing main tracker.")
                    main_tracker.update_with_detection(main_bbox, current_gray_frame)
        
        if main_tracker is not None and (frame_count % DETECTION_INTERVAL != 0):
            main_tracker.update(current_gray_frame)

        if main_tracker is not None:
            if not main_tracker.active:
                print(f"  Main tracker (ID: {main_tracker.tracker_id}) lost. Resetting.")
                main_tracker = None
            else:
                x, y, w, h = map(int, main_tracker.bbox)
                if w > 0 and h > 0:
                    color = (0, 255, 0) if main_tracker.missing_frames_count == 0 else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"Main Target ID: {main_tracker.tracker_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Single Object Tracking (Corrected)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Key 'q' pressed. Exiting.")
            break

        frame_count += 1
        
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")


if __name__ == "__main__":
    source_to_process = None
    while True:
        print("Please select the input source:")
        print("1: Live Webcam")
        print("2: Video File")
        choice = input("Your choice (1 or 2): ")

        if choice == '1':
            try:
                webcam_index_str = input("Enter webcam index number [default: 0]: ")
                source_to_process = int(webcam_index_str) if webcam_index_str.strip() else 0
                break
            except ValueError:
                print("Invalid index. Please enter a number.")

        elif choice == '2':
            video_path = input("Enter the full path to the video file: ").strip().strip('"')
            if os.path.exists(video_path):
                source_to_process = video_path
                break
            else:
                print(f"Error: File '{video_path}' does not exist. Please check the path.")
        
        else:
            print("Invalid choice. Please enter 1 or 2.")

    if source_to_process is not None:
        run_tracker(source=source_to_process)

    print("Program finished.")


