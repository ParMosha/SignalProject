import cv2

def show_camera_with_kcf(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open video source")
        return

    tracker = cv2.TrackerKCF_create()
    initBB = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if initBB is not None:
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failure", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 's' to select object to track", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('KCF Tracker', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to exit
            break
        elif key == ord('s') and initBB is None:
            initBB = cv2.selectROI('KCF Tracker', frame, fromCenter=False, showCrosshair=True)
            if initBB[2] > 0 and initBB[3] > 0:
                tracker.init(frame, initBB)
            else:
                initBB = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose video source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("Enter 1 or 2: ")
    if choice == '1':
        show_camera_with_kcf(0)
    elif choice == '2':
        path = input("Enter path to video file: ")
        show_camera_with_kcf(path)
    else:
        print("Invalid choice.")