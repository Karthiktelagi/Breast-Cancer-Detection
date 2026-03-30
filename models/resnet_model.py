import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet50(pretrained=True)

        # Change first layer (1 channel → 3 channel)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change final layer
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.model(x)