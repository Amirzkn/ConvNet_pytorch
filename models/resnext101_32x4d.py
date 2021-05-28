from __future__ import print_function, division, absolute_import
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .resnext_features import resnext101_32x4d_features
from .resnext_features import resnext101_64x4d_features

__all__ = ['ResNeXt101_32x4d', 'resnext101_32x4d',
           'ResNeXt101_64x4d', 'resnext101_64x4d']

model_urls = {
    'resnext101_32x4d':'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth',
    'resnext101_64x4d':'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth',
           
}

class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def resnext101_32x4d(pretrained=False, progress=True, **kwargs):
    model = ResNeXt101_32x4d(**kwargs)
    if pretrained:
       
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x4d']))       
    return model


