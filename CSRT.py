import cv2

def select_video_source():
    print("1. Use laptop camera")
    print("2. Use video file")
    choice = input("Please select an option (1 or 2): ")
    
    if choice == '1':
        return 0
    elif choice == '2':
        video_path = input("Please enter the full path to video file: ")
        return video_path
    else:
        print("Invalid option! Using camera by default.")
        return 0

tracker = cv2.TrackerCSRT_create()
video_source = select_video_source()
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame")
    exit()

bbox = cv2.selectROI("Select Object to Track", frame, False)
cv2.destroyWindow("Select Object to Track")

ret = tracker.init(frame, bbox)
if not ret:
    print("Error: Tracker initialization failed")
    exit()

fps = 0
frame_count = 0
start_time = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    
    frame_count += 1
    if frame_count >= 10:
        end_time = cv2.getTickCount()
        fps = 10 * cv2.getTickFrequency() / (end_time - start_time)
        start_time = end_time
        frame_count = 0
    
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        
        cv2.putText(frame, "CSRT Tracker", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, "Status: Tracking", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    cv2.imshow("CSRT Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()