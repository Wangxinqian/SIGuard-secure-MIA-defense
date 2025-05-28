import torch
import torch.nn as nn
import torch.nn.functional as F

class Mem_attacker(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.classifier = nn.Sequential(
          nn.Linear(self.input_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
    def forward(self, x):
        logits = self.classifier(x)
        return logits.flatten()

class chminist_target_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Flatten(start_dim=1),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 8)
        )
    def forward(self, x):
        output = self.classifier(x)
        return output