import cv2
import os

def select_video_source():
    print("1. Use laptop camera")
    print("2. Use video file")
    while True:
        choice = input("Please select an option (1 or 2): ")
        if choice == '1':
            return 0
        elif choice == '2':
            while True:
                video_path = input("Please enter the full path to video file: ").strip('"\'')
                video_path = os.path.normpath(video_path)
                if os.path.exists(video_path):
                    return video_path
                print(f"Error: File not found at {video_path}. Please try again.")
        else:
            print("Invalid option! Please enter 1 or 2.")

def select_object(frame):
    while True:
        bbox = cv2.selectROI("Select Object to Track", frame, False)
        cv2.destroyWindow("Select Object to Track")
        
        if bbox == (0, 0, 0, 0):
            print("Error: No object selected or invalid selection")
            continue
            
        if bbox[2] < 10 or bbox[3] < 10:
            print("Error: Selected area too small (minimum 10x10 pixels)")
            continue
            
        return bbox

def main():
    tracker = cv2.TrackerCSRT_create()
    video_source = select_video_source()
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        if video_source != 0:
            print("Please check:")
            print("1. The file exists")
            print("2. You have proper file permissions")
            print("3. The file is a supported video format")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video source")
        cap.release()
        return
    
    try:
        bbox = select_object(frame)
        
        if not tracker.init(frame, bbox):
            print("Error: Tracker initialization failed - trying alternative method")
            tracker = cv2.legacy.TrackerCSRT_create()
            if not tracker.init(frame, bbox):
                print("Error: Still unable to initialize tracker")
                cap.release()
                return
        
        fps = 0
        frame_count = 0
        start_time = cv2.getTickCount()
        video_loops = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                if video_source == 0: 
                    break
      
                video_loops += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
                continue
            
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
                if video_source != 0:
                    cv2.putText(frame, f"Loops: {video_loops}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failure", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press ESC to exit", (frame.shape[1]-200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
            cv2.imshow("CSRT Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()