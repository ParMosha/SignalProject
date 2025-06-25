import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms


def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Example usage:
if __name__ == "__main__":
    num_classes = 2  # 1 class (object) + background
    model = get_faster_rcnn_model(num_classes)
    print(model)
    image_path = "my_image.jpg"
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    model.eval()

    with torch.no_grad():
        prediction = model([image_tensor])

    print("Detection output:", prediction)