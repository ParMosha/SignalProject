import cv2
import torch
import urllib.request

# Path to manually downloaded weights
weights_url = "https://raw.githubusercontent.com/ParMosha/SignalProject/main/yolov5s.pt"
weights_path = '/your/readable/path/yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, source='local')

def detect_objects(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    detections = results.pandas().xyxy[0]
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        conf = row['confidence']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow('Detections', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_url = "https://raw.githubusercontent.com/ParMosha/SignalProject/main/9.jpg"
    image_path = "downloaded_image.jpg"
    urllib.request.urlretrieve(image_url, image_path)
    detect_objects(image_path)
