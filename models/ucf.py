"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: UCF detector (inference-only port)

Reference:
@article{yan2023ucf,
  title={UCF: Uncovering Common Features for Generalizable Deepfake Detection},
  author={Yan, Zhiyuan and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2304.13949},
  year={2023}
}

Ported from DeepfakeBench (CC BY-NC 4.0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .xception import Xception


# ── Helper modules ──────────────────────────────────────────────────────────


def r_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def c_norm(self, x, bs, ch, eps=1e-7):
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0) == y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = y.reshape(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out = ((x - x_mean.expand(size)) / x_std.expand(size)) * y_std.expand(
            size
        ) + y_mean.expand(size)
        return out


class Conditional_UNet(nn.Module):
    def __init__(self):
        super(Conditional_UNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)

        self.adain3 = AdaIN()
        self.adain2 = AdaIN()
        self.adain1 = AdaIN()

        self.dconv_up3 = r_double_conv(512, 256)
        self.dconv_up2 = r_double_conv(256, 128)
        self.dconv_up1 = r_double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, 3, 1)
        self.up_last = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.activation = nn.Tanh()

    def forward(self, c, x):
        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up3(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up3(c)

        x = self.adain2(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up2(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up2(c)

        x = self.adain1(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up1(x)

        x = self.conv_last(x)
        out = self.up_last(x)
        return self.activation(out)


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_f, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_f, 1, 1),
        )

    def forward(self, x):
        return self.conv2d(x)


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_f),
        )

    def forward(self, x):
        bs = x.size(0)
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat


class UCFDetector(nn.Module):
    ENCODER_FEAT_DIM = 512
    HALF_FINGERPRINT_DIM = 256
    SPECIFIC_TASK_NUMBER = 5
    NUM_CLASSES = 2

    def __init__(self):
        super().__init__()
        encoder_backbone_config = {
            "num_classes": self.NUM_CLASSES,
            "inc": 3,
            "dropout": False,
            "mode": "adjust_channel",
        }
        self.encoder_f = Xception(encoder_backbone_config)
        self.encoder_c = Xception(encoder_backbone_config)

        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.con_gan = Conditional_UNet()

        self.head_spe = Head(
            in_f=self.HALF_FINGERPRINT_DIM,
            hidden_dim=self.ENCODER_FEAT_DIM,
            out_f=self.SPECIFIC_TASK_NUMBER,
        )
        self.head_sha = Head(
            in_f=self.HALF_FINGERPRINT_DIM,
            hidden_dim=self.ENCODER_FEAT_DIM,
            out_f=self.NUM_CLASSES,
        )
        self.block_spe = Conv2d1x1(
            in_f=self.ENCODER_FEAT_DIM,
            hidden_dim=self.HALF_FINGERPRINT_DIM,
            out_f=self.HALF_FINGERPRINT_DIM,
        )
        self.block_sha = Conv2d1x1(
            in_f=self.ENCODER_FEAT_DIM,
            hidden_dim=self.HALF_FINGERPRINT_DIM,
            out_f=self.HALF_FINGERPRINT_DIM,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forgery_features = self.encoder_f.features(x)
        f_share = self.block_sha(forgery_features)
        out_sha, _ = self.head_sha(f_share)
        return out_sha
