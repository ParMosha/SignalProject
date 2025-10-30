import cv2
from ultralytics import YOLO
import urllib.request
import tempfile
import os
import matplotlib.pyplot as plt

def load_model(model_path):
    """
    Load the YOLO model from the specified path.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_objects(model, image_path):
    """
    Detect objects in the image using the YOLO model.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to read.")
        
        results = model(image)
        return results
    except Exception as e:
        print(f"Error detecting objects: {e}")
        return None

def download_image(url, save_path):
    """
    Download an image from a URL and save it to the specified path.
    """
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Image downloaded and saved to {save_path}")
    except Exception as e:
        print(f"Error downloading image: {e}")

def main():
    model_url = "https://raw.githubusercontent.com/ParMosha/SignalProject/main/yolo11s.pt"
    image_url = "https://raw.githubusercontent.com/ParMosha/SignalProject/main/image.png"
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, "downloaded_weights.pt")
        image_path = os.path.join(tmpdirname, "input_image.jpg")
        urllib.request.urlretrieve(model_url, model_path)
        urllib.request.urlretrieve(image_url, image_path)
        model = load_model(model_path)
        if model:
            results = detect_objects(model, image_path)
            if results:
                print("Detection results:", results)
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError("Image not found or unable to read.")
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
            
            else:
                print("No objects detected or an error occurred.")

if __name__ == "__main__":
    main()

