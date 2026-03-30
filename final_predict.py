import torch
import cv2
import numpy as np
import joblib
import torch.nn.functional as F

from models.cnn_model import CNNModel
from models.resnet_model import ResNetModel

classes = ["Normal", "Benign", "Malignant"]

# Load image
img = cv2.imread("test1.jpg", 0)
img = cv2.resize(img, (224, 224))
img = img / 255.0

img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

# ---------------- CNN ----------------
cnn = CNNModel()
cnn.load_state_dict(torch.load("model.pth", weights_only=True))
cnn.eval()

cnn_out = cnn(img_tensor)
cnn_pred = torch.argmax(cnn_out).item()

# ---------------- ResNet ----------------
resnet = ResNetModel()
resnet.load_state_dict(torch.load("resnet_model.pth", weights_only=True))
resnet.eval()

resnet_out = resnet(img_tensor)
resnet_pred = torch.argmax(resnet_out).item()

# ---------------- Hybrid ----------------
with torch.no_grad():
    features = cnn.conv(img_tensor)
    features = F.adaptive_avg_pool2d(features, (1, 1))
    features = features.view(1, -1).numpy()

svm = joblib.load("svm_model.pkl")
hybrid_pred = svm.predict(features)[0]

# ---------------- OUTPUT ----------------

print("\n🔍 FINAL RESULTS:\n")

print("CNN Prediction      :", classes[cnn_pred])
print("ResNet Prediction   :", classes[resnet_pred])
print("Hybrid Prediction   :", classes[hybrid_pred])