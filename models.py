import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.ensemble as se
import sklearn.calibration as sc
import torchvision.models as tm
from transformers import AutoModelForImageClassification
# from laplace import Laplace


def get_model(unc_method, base_model, num_classes=10):
    if unc_method == "ensemble":
        return get_base_model(base_model, num_classes)
    else:
        raise ValueError("Invalid uncertainty method")


def get_base_model(base_model, num_classes=10, use_dropout=False):
    if base_model == "cnn":
        return CNN(use_dropout)
    elif base_model == "lenet":
        return LeNet(use_dropout)
    elif base_model == "fcnet":
        return FCNet(use_dropout=use_dropout)
    elif base_model == "resnet":
        return ResNet(use_dropout=use_dropout, num_classes=num_classes)
    elif base_model == "efficientnet":
        return EfficientNet(use_dropout=use_dropout, num_classes=num_classes)
    elif base_model == "vit":
        return ViT(num_classes=num_classes, use_dropout=use_dropout)
    else:
        raise ValueError("Invalid model name")


class ViT(nn.Module):
    def __init__(self, num_classes=101, use_dropout=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.model = AutoModelForImageClassification.from_pretrained("nateraw/food")
        # self.model = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")

    def forward(self, x):
        x = self.model.vit(x)
        x = x[0]
        if self.use_dropout and not self.training:
            x = F.dropout(x, training=True)
        x = self.model.classifier(x[:, 0, :])
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(tm.ResNet):
    def __init__(self, num_classes=10, use_dropout=False):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])
        self.use_dropout = use_dropout
        # load pretrained weights from torchvision models
        # self.fc = nn.Linear(512, num_classes)

        self.load_state_dict(tm.ResNet18_Weights.DEFAULT.get_state_dict())
        self.fc = nn.Linear(512, num_classes)


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
        if self.use_dropout and not self.training:
            x = F.dropout(x, training=True)
        x = self.fc(x)

        return x

    def predict(self, x):
        return torch.unsqueeze(self.forward(x), 2)

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, use_dropout=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.model = tm.efficientnet_v2_s(weights=tm.EfficientNet_V2_S_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if self.use_dropout:
            x = F.dropout(x, training=True)
        x = self.model.classifier(x)
        return x

    def predict(self, x):
        return torch.unsqueeze(self.forward(x), 2)


class LeNet(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        # self.fc1 = nn.Linear(1250, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout and not self.training:
            x = F.dropout(x, training=True)
        x = self.fc2(x)
        return x

    def predict(self, x):
        return torch.unsqueeze(self.forward(x), 2)


class CNN(nn.Module):

    def __init__(self, use_dropout=False):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        # self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512)
        self.use_dropout = use_dropout

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=True)
        x = self.fc2(x)
        return x



class FCNet(nn.Module):
    def __init__(self, features_in=784, use_dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(features_in, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.use_dropout = use_dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=True)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = F.dropout(x, training=True)
        x = self.fc3(x)
        return x

    def predict(self, x):
        return torch.unsqueeze(self.forward(x), 2)


class Ensemble(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.members = []
        self.num_classes = num_classes

    def predict(self, x, samples=5):
        if samples > len(self.members):
            raise ValueError("Not enough members in ensemble")
        preds = torch.empty((x.shape[0], self.num_classes, samples), device=x.device)
        with torch.no_grad():
            for i in range(samples):
                preds[:, :, i] = F.softmax(self.members[i](x), dim=1)
        return preds


class ActiveEnsemble(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.members = [FCNet() for _ in range(num_models)]


    def forward(self, x):
        preds = torch.empty((x.shape[0], 10, len(self.members)))
        for i in range(len(self.members)):
            preds[:, :, i] = F.softmax(self.members[i](x), dim=1)
        return preds

    def predict(self, x):
        return self.forward(x)

