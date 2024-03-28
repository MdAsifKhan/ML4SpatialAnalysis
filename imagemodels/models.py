import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as models

from torchvision.models import vit_b_16

class ResNet50SingleChannel(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        super(ResNet50SingleChannel, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        
        # Change the first convolutional layer to accept one input channel
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if in_channels == 1:
            # Change the first layer's weight to be a single-channel version of the pretrained weights
            self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)
        else:
            weight = resnet.conv1.weight.data
            # For simplicity, duplicate the weights along the channel dimension
            self.conv1.weight.data = torch.cat([weight] * (in_channels // 3 + 1), dim=1)[:, :in_channels, :, :]

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class ResNetClassifier(nn.Module):
    def __init__(self, config):
        super(ResNetClassifier, self).__init__()
        self.config = config
        self.resnet = ResNet50SingleChannel(self.config['nm_markers'], pretrained=self.config['pretrained'])
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.final_layer = nn.Sequential(
                            nn.Linear(in_features, self.config['nm_markers']),
                            nn.LeakyReLU(0.2),
                            nn.Linear(self.config['nm_markers'], self.config['nm_classes'])
                        )

    def forward(self, x):
        batch_size, marker_images, W, H = x.size()
        # Reshape the input to (batch_size * marker_images, W, H)
        #x = x.view(-1, 1, W, H)
        features = self.resnet(x)
        #features = features.view(batch_size, -1)
        output = self.final_layer(features)
        return output


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        self.config = config
        self.vit = vit_b_16(self.config['pretrained'])
        in_features = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.final_layer = nn.Sequential(
                            nn.Linear(self.config['nm_markers']*in_features, self.config['nm_markers']),
                            nn.LeakyReLU(0.2),
                            nn.Linear(self.config['nm_markers'],  self.config['nm_classes'])
                    )

    def forward(self, x):
        batch_size, marker_images, C, W, H = x.size()
        # Reshape the input to (batch_size * marker_images, C, W, H)
        x = x.view(-1, C, W, H)
        features = self.vit(x)
        features = features.view(batch_size, -1)
        output = self.final_layer(features)
        return output