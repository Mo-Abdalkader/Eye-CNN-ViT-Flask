import torch  # 2.6.0+cu124
import timm  # For ViT
import torch.nn as nn
from torchvision import models


class DualOutputResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        self.bn = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.5)
        self.shared_dense = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )

        self.ret_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

        self.mac_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        x = self.dropout(x)

        shared = self.shared_dense(x)
        ret_out = self.ret_head(shared)
        mac_out = self.mac_head(shared)

        return ret_out, mac_out


class DualOutputViT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Identity()
        self.bn = nn.BatchNorm1d(768)
        self.dropout = nn.Dropout(0.5)
        self.shared_dense = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )

        self.ret_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

        self.mac_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.bn(x)
        x = self.dropout(x)

        shared = self.shared_dense(x)
        ret_out = self.ret_head(shared)
        mac_out = self.mac_head(shared)

        return ret_out, mac_out


def create_dual_output_model(model_type, device):
    if model_type == 'ResNet50':
        model = DualOutputResNet50(pretrained=True)

    elif model_type == 'ViT':
        model = DualOutputViT(pretrained=True)

    else:
        raise ValueError("model_type must be 'ResNet50' or 'ViT'")

    return model.to(device)
