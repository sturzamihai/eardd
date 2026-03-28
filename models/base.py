"""
Factory functions for loading pretrained deepfake detectors.

Each loader returns an nn.Module with:
  - forward(x: Tensor) -> Tensor   # (B,3,H,W) -> logits (B,2)
  - eval() mode, moved to the requested device
"""

import torch
import torch.nn as nn

from .xception import XceptionDetector
from .efficientnet import EfficientDetector
from .ucf import UCFDetector
from .recce import RecceDetector


def _load(model: nn.Module, path: str, strict: bool = True) -> nn.Module:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt, strict=strict)
    return model


def load_xception(path: str, device: str = 'cpu') -> XceptionDetector:
    model = XceptionDetector()
    _load(model, path)
    return model.to(device).eval()


def load_effnetb4(path: str, device: str = 'cpu') -> EfficientDetector:
    model = EfficientDetector()
    _load(model, path)
    return model.to(device).eval()


def load_ucf(path: str, device: str = 'cpu') -> UCFDetector:
    model = UCFDetector()
    _load(model, path)
    return model.to(device).eval()


def load_recce(path: str, device: str = 'cpu') -> RecceDetector:
    model = RecceDetector()
    _load(model, path)
    return model.to(device).eval()


MODELS = {
    'xception': load_xception,
    'effnetb4': load_effnetb4,
    'ucf': load_ucf,
    'recce': load_recce,
}
