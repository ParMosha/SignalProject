import cv2
import time
import numpy as np

def show_camera_with_kcf(source=0):
    cap = cv2.VideoCapture(source)
    tracker = cv2.TrackerKCF_create()
    initBB = None

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read from video source")
        return
    
    cv2.putText(first_frame, "Select object to track, then press SPACE to continue", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('KCF Tracker', first_frame)
    
    print("Select the object you want to track in the video window, then press SPACE to start tracking")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord(' '): 
            initBB = cv2.selectROI('KCF Tracker', first_frame, fromCenter=False, showCrosshair=True)
            if initBB[2] > 0 and initBB[3] > 0:
                tracker.init(first_frame, initBB)
                break
            else:
                print("Invalid selection. Try again by pressing SPACE")

    times = []
    max_samples = 30
    prev_time = time.time()
    avg_fps = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        times.append(curr_time)
        if len(times) > max_samples:
            times.pop(0)
        if len(times) > 1:
            intervals = np.diff(np.array(times))
            mean_interval = np.mean(intervals)
            if mean_interval > 0:
                avg_fps = 1.0 / mean_interval
            else:
                avg_fps = 0.0
        else:
            avg_fps = 0.0

        if initBB is not None:
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failure", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('KCF Tracker', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose input source:")
    print("1. Webcam (real-time)")
    print("2. Video file")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        show_camera_with_kcf(0)
    elif choice == "2":
        video_path = input("Enter path to video file: ").strip()
        show_camera_with_kcf(video_path)
    else:
        print("Invalid choice.")
