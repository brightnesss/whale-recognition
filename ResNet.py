import torch
import torch.nn as nn
import torchvision.models as models
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class TripletsModel(nn.Module):
    def __init__(self, embedding_size, num_classes, pretrained=False):
        super(TripletsModel, self).__init__()

        resnet50 = models.resnet50(pretrained)

        self.embedding_size = embedding_size

        self.pretrain = nn.Sequential(*list(resnet50.children())[:-1])

        self.fc = nn.Linear(2048, self.embedding_size)

        self.classifier = nn.Linear(self.embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.pretrain(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        self.features = self.features * alpha

        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        return res
