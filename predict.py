import torch
import cv2
from models.cnn_model import CNNModel

# Load model
model = CNNModel()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

classes = ["Normal", "Benign", "Malignant"]

# Load image
img_path = "test2.jpg"

img = cv2.imread(img_path, 0)
img = cv2.resize(img, (224, 224))
img = img / 255.0

img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

# Prediction
output = model(img)
probs = torch.softmax(output, dim=1)

print("\nClass Probabilities:")
for i, cls in enumerate(classes):
    print(f"{cls}: {probs[0][i].item():.4f}")

pred = torch.argmax(probs).item()

print("\n✅ Final Prediction:", classes[pred])