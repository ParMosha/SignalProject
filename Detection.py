import cv2
import torch
from ultralytics import YOLO
import urllib.request

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
    model_path = "downloaded_weights.pt"
    urllib.request.urlretrieve(model_url, model_path)
    image_url = "https://raw.githubusercontent.com/ParMosha/SignalProject/main/image.png"
    image_path = 'input_image.jpg'  # Path to the input image
    urllib.request.urlretrieve(image_url, image_path)
    model = load_model(model_path)
    if model:
        results = detect_objects(model, image_path)
        if results:
            print("Detection results:", results)
        else:
            print("No objects detected or an error occurred.")
if __name__ == "__main__":
    main()
