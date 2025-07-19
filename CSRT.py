import cv2

tracker = cv2.TrackerCSRT_create()

# استفاده از دوربین لپتاپ (0) یا مسیر فایل ویدیویی
video_source = 0  # یا مسیر فایل مانند 'video.mp4'
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
        
        cv2.putText(frame, f"CSRT Tracker", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: Tracking", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    cv2.imshow("CSRT Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()