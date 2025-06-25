import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def detect_objects(model, image, threshold=0.7):
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
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

def visualize_detections(image, detections):
    _, ax = plt.subplots(1)
    ax.imshow(image)
    for det in detections:
        x1, y1, x2, y2 = det['box']
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none',alpha=0.5)
        ax.add_patch(rect)
        ax.text(x1, y1, f'Label: {det["label"]}, Score: {det["score"]:.2f}',
                bbox=dict(facecolor='yellow', alpha=0.1), fontsize=8, color='black')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # num_classes = int(input("Enter number of classes (including background): "))
    num_classes = 2
    model = get_faster_rcnn_model(num_classes)
    image_url = "https://raw.githubusercontent.com/ParMosha/SignalProject/main/9.jpg"
    image = load_image_from_url(image_url)
    threshold = 0.5
    detections, prediction = detect_objects(model, image, threshold)

    print("Detection output:", prediction)
    for det in detections:
        print(f"Detected object with label {det['label']} at {det['box']} with confidence {det['score']:.2f}")

    visualize_detections(image, detections)
