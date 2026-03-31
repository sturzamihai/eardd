from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_metric


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    lbls = labels.cpu().numpy()
    acc = (preds == lbls).mean()
    try:
        auc = roc_auc_score(lbls, probs)
    except ValueError:
        auc = float("nan")
    return acc, auc


def attack_success_rate(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    labels: np.ndarray,
) -> float:
    mask = (labels == 1) & (clean_preds == 1)
    if mask.sum() == 0:
        return float("nan")
    return float((adv_preds[mask] == 0).mean())


def class_acc(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    real, fake = labels == 0, labels == 1
    r = float((preds[real] == labels[real]).mean()) if real.any() else float("nan")
    f = float((preds[fake] == labels[fake]).mean()) if fake.any() else float("nan")
    return r, f


def video_level_auc(
    probs: np.ndarray, labels: np.ndarray, video_ids: list[str]
) -> float:
    vid_scores: dict[str, list[float]] = defaultdict(list)
    vid_label: dict[str, int] = {}
    for p, l, v in zip(probs, labels, video_ids):
        vid_scores[v].append(float(p))
        vid_label[v] = int(l)
    vids = list(vid_scores)
    try:
        return roc_auc_score(
            [vid_label[v] for v in vids], [np.mean(vid_scores[v]) for v in vids]
        )
    except ValueError:
        return float("nan")


def compute_perturbation_metrics(
    orig: torch.Tensor,
    adv: torch.Tensor,
    lpips_fn,
    lpips_chunk: int = 32,
) -> dict[str, float]:
    orig_01 = (orig + 1) / 2
    adv_01 = (adv + 1) / 2

    l2 = (orig_01 - adv_01).norm(2, dim=(1, 2, 3)).mean().item()

    mse = ((orig_01 - adv_01) ** 2).mean(dim=(1, 2, 3))
    psnr = (10 * torch.log10(1.0 / mse.clamp(min=1e-10))).mean().item()

    ssim = ssim_metric(
        adv_01, orig_01, data_range=1.0, reduction="elementwise_mean"
    ).item()

    lpips_scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(orig), lpips_chunk):
            scores = lpips_fn(orig[i : i + lpips_chunk], adv[i : i + lpips_chunk])
            lpips_scores.extend(scores.flatten().cpu().tolist())

    return {
        "L2": l2,
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": float(np.mean(lpips_scores)),
    }
