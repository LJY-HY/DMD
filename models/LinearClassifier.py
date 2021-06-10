import torch
import torch.nn as nn
import torch.nn.functional as F

model_dict = {
    'MobileNet' : 1280,
    'Inception' : 2048,
    'ShuffleNet' : 1024
}

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='Inception', num_classes=13):
        super(LinearClassifier, self).__init__()
        feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)