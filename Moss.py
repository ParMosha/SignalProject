import cv2
import numpy as np
import time

class MOSSETracker:
    def __init__(self, learning_rate=0.125):
        self.learning_rate = learning_rate
        self.filter = None
        self.gaussian = None
        self.window_size = None
        self.position = None
        self.eps = 1e-5

    def _create_gaussian_response(self, size):
        h, w = size
        sigma = h / 8
        y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
        return np.exp(-(x**2 + y**2) / (2 * sigma**2))

    def _preprocess(self, image):
        processed = np.log(image + 1)
        processed = (processed - processed.mean()) / (processed.std() + self.eps)
        window = np.outer(
            np.hanning(processed.shape[0]),
            np.hanning(processed.shape[1])
        )
        return processed * window

    def initialize(self, frame, bounding_box):
        x, y, w, h = bounding_box
        self.window_size = (w, h)
        self.position = (x + w//2, y + h//2)

        target = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        target = target.astype(np.float32)

        self.gaussian = np.fft.fft2(self._create_gaussian_response((h, w)))

        pre = self._preprocess(target)
        T = np.fft.fft2(pre)
        A = self.gaussian * np.conj(T)
        B = T * np.conj(T) + self.eps
        self.filter = A / B

        for _ in range(8):
            angle = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.95, 1.05)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
            warped = cv2.warpAffine(target, M, (w, h))

            Tw = np.fft.fft2(self._preprocess(warped))
            A_aug = self.gaussian * np.conj(Tw)
            B_aug = Tw * np.conj(Tw) + self.eps
            H_aug = A_aug / B_aug

            self.filter = (1 - self.learning_rate) * self.filter \
                          + self.learning_rate * H_aug

    def update(self, frame):
        if self.filter is None:
            raise RuntimeError("Tracker has not been initialized.")

        w, h = self.window_size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        patch = cv2.getRectSubPix(
            gray, (w, h), self.position
        )
        P = np.fft.fft2(self._preprocess(patch))

        resp = np.fft.ifft2(self.filter * P)
        resp = np.real(resp)

        _, _, _, (mx, my) = cv2.minMaxLoc(resp)
        dx = mx - w//2
        dy = my - h//2

        self.position = (self.position[0] + dx,
                         self.position[1] + dy)

        cx, cy = int(self.position[0] - w//2), int(self.position[1] - h//2)
        new_patch = gray[cy:cy+h, cx:cx+w]

        if new_patch.shape == (h, w):
            Tn = np.fft.fft2(self._preprocess(new_patch))
            A_n = self.gaussian * np.conj(Tn)
            B_n = Tn * np.conj(Tn) + self.eps
            Hn = A_n / B_n
            self.filter = (1 - self.learning_rate) * self.filter \
                          + self.learning_rate * Hn

        return (cx, cy, w, h)

def get_video_source():
    print("1. Use camera")
    print("2. Use video file")
    choice = input("Enter choice (1 or 2): ").strip()
    if choice == "2":
        return input("Enter video file path: ").strip()
    return 0

if __name__ == "__main__":
    video_source = get_video_source()
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: could not open video source.")
        exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: could not read first frame.")
        exit(1)

    bbox = cv2.selectROI("Select Object", first_frame, False, False)
    cv2.destroyWindow("Select Object")

    if bbox == (0, 0, 0, 0):
        print("No ROI selected, exiting.")
        exit(1)

    tracker = MOSSETracker(learning_rate=0.1)
    tracker.initialize(first_frame, bbox)

    frame_count = 0
    start_time = time.time()
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()
        
        x, y, w, h = tracker.update(frame)
        
        frame_end = time.time()
        frame_time = frame_end - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_list.append(current_fps)
        frame_count += 1
        
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        
        cv2.putText(frame, f"Current FPS: {current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Average FPS: {avg_fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("MOSSE Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key to exit
            break


    total_time = time.time() - start_time
    overall_avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nTracking completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Overall average FPS: {overall_avg_fps:.2f}")
    print(f"Processing average FPS: {sum(fps_list) / len(fps_list):.2f}" if fps_list else "N/A")

    cap.release()
    cv2.destroyAllWindows()



