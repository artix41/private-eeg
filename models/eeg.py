import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models


class EEG_CNN(nn.Module):
    def __init__(self):
        super(EEG_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64*5*5, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        x = x.view((-1, 3, 224, 224))
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view((-1, 64*5*5))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



# def EEG_CNN():
#     model = models.alexnet(pretrained=True)
#     model.classifier[6] = nn.Linear(4096, 7)
#
#     return model

