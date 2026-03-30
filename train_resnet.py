import torch
from torch.utils.data import DataLoader
from utils.dataset import BreastDataset
from models.resnet_model import ResNetModel

dataset = BreastDataset("data/train")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = ResNetModel()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = (correct / total) * 100

    print(f"ResNet Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.2f}%")

torch.save(model.state_dict(), "resnet_model.pth")
