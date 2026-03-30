import torch
import numpy as np
import joblib
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from models.cnn_model import CNNModel
from utils.dataset import BreastDataset
import torch.nn.functional as F

# Load dataset
dataset = BreastDataset("data/train")
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# Load trained CNN
model = CNNModel()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

features = []
labels = []

# Extract features
with torch.no_grad():
    for imgs, lbls in loader:
        x = model.conv(imgs)

        # 🔥 Reduce feature size (IMPORTANT)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        features.append(x.numpy())
        labels.append(lbls.numpy())

# Convert to numpy
features = np.vstack(features)
labels = np.hstack(labels)

print("Feature shape:", features.shape)

# Train SVM
svm = SVC(kernel='rbf', probability=True)
svm.fit(features, labels)

# Save model
joblib.dump(svm, "svm_model.pkl")

print("✅ Hybrid model trained & saved!")