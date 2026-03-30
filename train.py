import torch
from torch.utils.data import DataLoader
from models.cnn_model import CNNModel
from utils.dataset import BreastDataset
from collections import Counter

# Load dataset
dataset = BreastDataset("data/train")

# 🔥 Show class distribution
counts = Counter(dataset.labels)
print("Class Distribution:", counts)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = CNNModel()

# 🔥 Class weights (IMPORTANT FIX)
total = sum(counts.values())
weights = []

for i in range(3):
    weights.append(total / counts[i])

weights = torch.tensor(weights, dtype=torch.float)

loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🔥 Increase training
epochs = 20

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total_samples = 0

    for imgs, labels in loader:
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    acc = (correct / total_samples) * 100

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), "model.pth")

print("✅ Training complete and model saved!")