import cv2
import numpy as np

class AdvancedCSRTTracker:
    def __init__(self, learning_rate=0.02, padding=2.0, sigma=0.2, lambda_=0.01):
        self.tracker = cv2.TrackerCSRT_create()
        
        # Set parameters (these are similar to the ones in the paper)
        self.tracker.setInitialLearningRate(learning_rate)
        self.tracker.setPadding(padding)
        self.tracker.setSigma(sigma)  # Gaussian kernel bandwidth
        self.tracker.setKernelLambda(lambda_)  # Regularization
        
    def init(self, frame, bbox):
        """Initialize tracker with first frame and bounding box"""
        return self.tracker.init(frame, bbox)
    
    def update(self, frame):
        """Update tracker with new frame"""
        success, bbox = self.tracker.update(frame)
        return success, bbox
    
    def get_channel_weights(self):
        """Get the channel reliability weights (if available)"""
        # Note: OpenCV's implementation doesn't expose all internal parameters directly
        # This would require a custom implementation
        pass
    
    def get_spatial_reliability_map(self):
        """Get the spatial reliability map (if available)"""
        # Note: OpenCV's implementation doesn't expose all internal parameters directly
        # This would require a custom implementation
        pass

# Usage example
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Webcam
    
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        exit()
    
    # Select ROI
    roi = cv2.selectROI("Select Object", frame, False, False)
    cv2.destroyWindow("Select Object")
    
    # Initialize our advanced tracker
    tracker = AdvancedCSRTTracker(learning_rate=0.03, sigma=0.1)
    tracker.init(frame, roi)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Advanced CSRT Tracking", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Advanced CSRT Tracker", frame)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()