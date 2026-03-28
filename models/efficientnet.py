"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is for EfficientNetB4 backbone. Ported from DeepfakeBench (CC BY-NC 4.0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNetB4(nn.Module):
    def __init__(self, efficientnetb4_config):
        super(EfficientNetB4, self).__init__()
        self.num_classes = efficientnetb4_config["num_classes"]
        inc = efficientnetb4_config["inc"]
        self.dropout = efficientnetb4_config["dropout"]
        self.mode = efficientnetb4_config["mode"]

        pretrained_path = efficientnetb4_config.get("pretrained")
        if pretrained_path:
            self.efficientnet = EfficientNet.from_pretrained(
                "efficientnet-b4", weights_path=pretrained_path
            )
        else:
            self.efficientnet = EfficientNet.from_name("efficientnet-b4")

        self.efficientnet._conv_stem = nn.Conv2d(
            inc, 48, kernel_size=3, stride=2, bias=False
        )
        self.efficientnet._fc = nn.Identity()

        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        self.last_layer = nn.Linear(1792, self.num_classes)

        if self.mode == "adjust_channel":
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1792, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, x):
        x = self.efficientnet.extract_features(x)
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout_layer(x)
        self.last_emb = x
        return self.last_layer(x)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EfficientDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone_config = {
            "num_classes": 2,
            "inc": 3,
            "dropout": False,
            "mode": "Original",
            "pretrained": None,
        }
        self.backbone = EfficientNetB4(backbone_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        return self.backbone.classifier(features)
