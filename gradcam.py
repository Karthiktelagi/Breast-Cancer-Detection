import torch
import cv2
import numpy as np
from models.cnn_model import CNNModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load model
model = CNNModel()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

# Target layer (last conv layer)
target_layer = model.conv[-1]

cam = GradCAM(model=model, target_layers=[target_layer])

# Load image (GRAYSCALE)
img = cv2.imread("test.jpg", 0)
img = cv2.resize(img, (224, 224))

# Normalize
img_norm = img / 255.0

# Convert to tensor (1 channel for model)
input_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).float()

# Generate CAM
grayscale_cam = cam(input_tensor=input_tensor)[0]

# 🔥 FIX: Convert grayscale → 3 channel for visualization
img_3ch = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)

# Overlay heatmap
visualization = show_cam_on_image(img_3ch, grayscale_cam, use_rgb=True)

# Show result
cv2.imshow("Grad-CAM", visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()