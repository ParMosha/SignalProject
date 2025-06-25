import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO


def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Example usage:
if __name__ == "__main__":
    num_classes = 2  # 1 class (object) + background
    model = get_faster_rcnn_model(num_classes)
    # print(model)
    # image_url = "https://raw.githubusercontent.com/ParMosha/SignalProject/main/my_image.jpg"
    image_url = "https://raw.githubusercontent.com/ParMosha/SignalProject/main/9.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    model.eval()

    with torch.no_grad():
        prediction = model([image_tensor])

    print("Detection output:", prediction)
    # The output will contain boxes, labels, and scores for detected objects
    for element in prediction[0]['boxes']:
        print("Box coordinates:", element.tolist())
    for element in prediction[0]['labels']:
        print("Label:", element.item())
    for element in prediction[0]['scores']:
        print("Score:", element.item())
    # Note: The scores are between 0 and 1, indicating the confidence of the
    # detection. A threshold can be applied to filter out low-confidence detections.
    # For example, you can set a threshold of 0.5:
    threshold = 0.7
    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        if score > threshold:
            print(f"Detected object with label {label.item()} at {box.tolist()} with confidence {score.item()}")
    # This will print the detected objects with their labels, bounding box coordinates, and confidence scores
    # You can visualize the results using libraries like matplotlib or OpenCV if needed.
    # For example, to visualize the image with bounding boxes:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
            if score > threshold:
                x1, y1, x2, y2 = box.tolist()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f'Label: {label.item()}, Score: {score.item():.2f}', 
                        bbox=dict(facecolor='yellow', alpha=0.5), fontsize=8, color='black')

        plt.axis('off')
        plt.show()