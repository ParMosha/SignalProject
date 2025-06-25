import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def get_faster_rcnn_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def detect_objects(model: torch.nn.Module, image: Image.Image, threshold: float = 0.7, device: str = "cpu"):
    if image is None:
            return [], {}
    if image.mode != "RGB":
            image = image.convert("RGB")
        # Manual conversion instead of transforms.ToTensor()
    image_np = np.array(image).astype(np.float32) / 255.0  # shape (H, W, C)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)  # shape (C, H, W)
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    results = []
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > threshold:
            results.append({
                'box': box.tolist(),
                'label': label.item(),
                'score': score.item()
            })
    return results, prediction

def visualize_detections(image: Image.Image, detections):
    _, ax = plt.subplots(1)
    ax.imshow(np.array(image))
    for det in detections:
        x1, y1, x2, y2 = det['box']
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        ax.text(x1, y1, f'Label: {det["label"]}, Score: {det["score"]:.2f}',
                bbox=dict(facecolor='yellow', alpha=0.1), fontsize=8, color='black')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2  # Including background
    model = get_faster_rcnn_model(num_classes).to(device)
    image_url = "https://raw.githubusercontent.com/ParMosha/SignalProject/main/9.jpg"
    image = load_image_from_url(image_url)
    threshold = 0.5
    detections, prediction = detect_objects(model, image, threshold, device)
    print("Detection output:", prediction)
    for det in detections:
        print(f"Detected object with label {det['label']} at {det['box']} with confidence {det['score']:.2f}")
    if image is not None:
        visualize_detections(image, detections)
