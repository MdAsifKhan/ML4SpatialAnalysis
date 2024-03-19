import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    def __init__(self, config):
        super(ResNetClassifier, self).__init__()
        self.config = config
        self.resnet = models.resnet18(pretrained=self.config['pretrained'])
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.final_layer = nn.Linear(self.config['nm_markers']*in_features, 1)

    def forward(self, x):
        batch_size, marker_images, C, W, H = x.size()
        # Reshape the input to (batch_size * marker_images, C, W, H)
        x = x.view(-1, C, W, H)
        features = self.resnet(x)
        features = features.view(batch_size, -1)
        output = self.final_layer(features)
        return output