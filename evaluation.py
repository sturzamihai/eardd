import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import compute_perturbation_metrics

MAX_LPIPS_SAMPLES = 500

_NAN4 = {"L2": float("nan"), "PSNR": float("nan"),
         "SSIM": float("nan"), "LPIPS": float("nan")}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    all_logits, all_labels, all_vids = [], [], []
    for images, labels, video_ids in loader:
        all_logits.append(model(images.to(device)).cpu())
        all_labels.append(labels)
        all_vids.extend(video_ids)
    return torch.cat(all_logits), torch.cat(all_labels), all_vids


def evaluate_on_adversarial(
    model: nn.Module,
    attack,
    loader: DataLoader,
    device: torch.device,
    lpips_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], dict[str, float]]:
    adv_logits_list, clean_preds_list, labels_list, vid_list = [], [], [], []
    orig_buf: list[torch.Tensor] = []
    adv_buf: list[torch.Tensor] = []
    n_collected = 0

    model.eval()
    for images, labels, video_ids in loader:
        images = images.to(device)
        labels_d = labels.to(device)

        adv_images = images.clone()
        fake_mask = labels_d == 1
        if fake_mask.any():
            adv_images[fake_mask] = attack(images[fake_mask], labels_d[fake_mask])
            if lpips_fn is not None and n_collected < MAX_LPIPS_SAMPLES:
                remaining = MAX_LPIPS_SAMPLES - n_collected
                orig_buf.append(images[fake_mask][:remaining].detach())
                adv_buf.append(adv_images[fake_mask][:remaining].detach())
                n_collected += min(int(fake_mask.sum()), remaining)

        with torch.no_grad():
            clean_preds_list.append(model(images).argmax(1).cpu())
            adv_logits_list.append(model(adv_images).cpu())
        labels_list.append(labels)
        vid_list.extend(video_ids)

    if lpips_fn is not None and orig_buf:
        perturb = compute_perturbation_metrics(
            torch.cat(orig_buf), torch.cat(adv_buf), lpips_fn
        )
    else:
        perturb = dict(_NAN4)

    return (
        torch.cat(adv_logits_list),
        torch.cat(clean_preds_list),
        torch.cat(labels_list),
        vid_list,
        perturb,
    )
