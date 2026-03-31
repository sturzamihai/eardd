"""
Adversarial robustness benchmark for deepfake detectors on Celeb-DF v2.

Attacks:
  White-box : FGSM (ε=0.05,0.1,2,8/255)  |  PGD-50 (ε=8/255)  |  C&W L2
  Black-box  : Square Attack (ε=8/255)
  Transfer   : PGD-50 adversarial examples from each source model
               evaluated on all target models → 4x4 ASR and video-AUC matrices

Metrics per attack: ACC (overall / real / fake), AUC (frame + video), ASR,
                    L2, PSNR, SSIM, LPIPS

Usage:
  python main.py \\
    --checkpoint-dir models/weights \\
    --data-dir       data/Celeb-DF-v2 \\
    --device         auto \\
    --models         xception effnetb4 ucf recce \\
    --batch-size     16 \\
    --cw-steps       200 \\
    --cw-samples     1000 \\
    --square-samples 1000
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import lpips
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchattacks

from dataset import CelebDFv2Dataset
from evaluation import evaluate, evaluate_on_adversarial
from metrics import (
    attack_success_rate,
    class_acc,
    compute_metrics,
    video_level_auc,
)
from models import MODELS


def get_device(override: str | None = None) -> torch.device:
    if override and override != "auto":
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD = [0.5, 0.5, 0.5]


def make_attack(attack):
    attack.set_normalization_used(mean=_NORM_MEAN, std=_NORM_STD)
    return attack


class RandomNoise:
    def __init__(self, eps: float, std: float = 0.5):
        self.eps_norm = eps / std

    def __call__(self, images: torch.Tensor, _labels: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(images).uniform_(-self.eps_norm, self.eps_norm)
        return (images + noise).clamp(-1.0, 1.0)


CHECKPOINT_NAMES = {
    "xception": "xception_best.pth",
    "effnetb4": "effnb4_best.pth",
    "ucf": "ucf_best.pth",
    "recce": "recce_best.pth",
}


def fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def print_table(rows: list[dict], title: str = ""):
    if not rows:
        return
    if title:
        print(f'\n{"─" * 70}')
        print(f"  {title}")
        print(f'{"─" * 70}')
    keys = list(rows[0].keys())
    col_w = {k: max(len(k), max(len(fmt(r[k])) for r in rows)) for k in keys}
    header = "  ".join(k.ljust(col_w[k]) for k in keys)
    print(header)
    print("  ".join("-" * col_w[k] for k in keys))
    for row in rows:
        print("  ".join(fmt(row[k]).ljust(col_w[k]) for k in keys))


def parse_args():
    p = argparse.ArgumentParser(description="Deepfake detector adversarial benchmark")
    p.add_argument(
        "--checkpoint-dir",
        default="models/weights",
        help="Directory containing *_best.pth files",
    )
    p.add_argument(
        "--data-dir", default="data/Celeb-DF-v2", help="Root of Celeb-DF v2 dataset"
    )
    p.add_argument("--output-dir", default="results", help="Where to save CSV results")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument(
        "--models",
        nargs="+",
        default=["xception", "effnetb4", "ucf", "recce"],
        choices=["xception", "effnetb4", "ucf", "recce"],
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--cw-steps",
        type=int,
        default=200,
        help="C&W optimisation steps (use 1000 for final results)",
    )
    p.add_argument(
        "--cw-samples",
        type=int,
        default=None,
        help="Max frames to evaluate for C&W (default: full dataset)",
    )
    p.add_argument(
        "--square-samples",
        type=int,
        default=None,
        help="Max frames to evaluate for Square Attack (default: full dataset)",
    )
    return p.parse_args()


def _attack_row(
    name: str,
    atk_name: str,
    adv_logits: torch.Tensor,
    cp: torch.Tensor,
    adv_labels: torch.Tensor,
    vid_ids: list[str],
    perturb: dict[str, float],
) -> dict:
    preds = adv_logits.argmax(1).numpy()
    acc, auc = compute_metrics(adv_logits, adv_labels)
    r_acc, f_acc = class_acc(preds, adv_labels.numpy())
    v_auc = video_level_auc(
        torch.softmax(adv_logits, 1)[:, 1].numpy(), adv_labels.numpy(), vid_ids
    )
    asr = attack_success_rate(cp.numpy(), preds, adv_labels.numpy())
    return {
        "model": name,
        "attack": atk_name,
        "ACC": acc,
        "real_ACC": r_acc,
        "fake_ACC": f_acc,
        "AUC": auc,
        "video_AUC": v_auc,
        "ASR": asr,
        **perturb,
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CelebDFv2Dataset(args.data_dir)
    print(f"Dataset: {len(dataset)} frames")

    def make_loader(max_samples: int | None) -> DataLoader:
        ds = dataset
        if max_samples is not None and max_samples < len(dataset):
            ds = torch.utils.data.Subset(dataset, range(max_samples))
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    loader = make_loader(None)

    models: dict[str, nn.Module] = {}
    for name in args.models:
        ckpt_path = Path(args.checkpoint_dir) / CHECKPOINT_NAMES[name]
        print(f"Loading {name} from {ckpt_path} ...", end=" ", flush=True)
        models[name] = MODELS[name](str(ckpt_path), device=str(device))
        print("ok")

    print("Loading LPIPS (AlexNet) ...", end=" ", flush=True)
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    print("ok")

    baseline_rows = []
    baseline_results: dict[str, dict] = {}
    for name, model in models.items():
        logits, labels, video_ids = evaluate(model, loader, device)
        acc, auc = compute_metrics(logits, labels)
        preds = logits.argmax(1).numpy()
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        r_acc, f_acc = class_acc(preds, labels.numpy())
        v_auc = video_level_auc(probs, labels.numpy(), video_ids)
        baseline_results[name] = {
            "logits": logits,
            "labels": labels,
            "preds": preds,
            "video_ids": video_ids,
        }
        baseline_rows.append(
            {
                "model": name,
                "ACC": acc,
                "real_ACC": r_acc,
                "fake_ACC": f_acc,
                "AUC": auc,
                "video_AUC": v_auc,
            }
        )
    print_table(baseline_rows, "Baseline (clean)")

    cw_loader = make_loader(args.cw_samples)
    if args.cw_samples is not None and args.cw_samples < len(dataset):
        print(f"C&W capped at {args.cw_samples} frames")

    wb_rows = []
    for name, model in models.items():
        cheap_attacks = [
            ("Random-0.1", RandomNoise(eps=0.1 / 255)),
            ("FGSM-0.05", make_attack(torchattacks.FGSM(model, eps=0.05 / 255))),
            ("FGSM-0.1", make_attack(torchattacks.FGSM(model, eps=0.1 / 255))),
            ("FGSM-2", make_attack(torchattacks.FGSM(model, eps=2 / 255))),
            ("FGSM-8", make_attack(torchattacks.FGSM(model, eps=8 / 255))),
            (
                "PGD-50",
                make_attack(
                    torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=50)
                ),
            ),
        ]

        for atk_name, attack in cheap_attacks:
            print(f"  {name} / {atk_name} ...", end=" ", flush=True)
            adv_logits, cp, adv_labels, vid_ids, perturb = evaluate_on_adversarial(
                model, attack, loader, device, lpips_fn
            )
            row = _attack_row(
                name, atk_name, adv_logits, cp, adv_labels, vid_ids, perturb
            )
            wb_rows.append(row)
            print(
                f"ACC={row['ACC']:.4f}  AUC={row['AUC']:.4f}  "
                f"ASR={row['ASR']:.4f}  PSNR={row['PSNR']:.1f}"
            )

        print(f"  {name} / CW-L2 ...", end=" ", flush=True)
        cw_attack = make_attack(torchattacks.CW(model, c=1.0, steps=args.cw_steps))
        cw_logits, cw_cp, cw_labels, cw_vids, cw_perturb = evaluate_on_adversarial(
            model, cw_attack, cw_loader, device, lpips_fn
        )
        row = _attack_row(
            name, "CW-L2", cw_logits, cw_cp, cw_labels, cw_vids, cw_perturb
        )
        wb_rows.append(row)
        print(
            f"ACC={row['ACC']:.4f}  AUC={row['AUC']:.4f}  "
            f"ASR={row['ASR']:.4f}  PSNR={row['PSNR']:.1f}"
        )

    print_table(wb_rows, "White-box attacks")

    square_loader = make_loader(args.square_samples)
    if args.square_samples is not None and args.square_samples < len(dataset):
        print(f"Square Attack capped at {args.square_samples} frames")

    bb_rows = []
    for name, model in models.items():
        print(f"  {name} / Square ...", end=" ", flush=True)
        sq_attack = make_attack(
            torchattacks.Square(
                model, norm="Linf", eps=8 / 255, n_queries=5000, p_init=0.05
            )
        )
        adv_logits, sq_cp, sq_labels, sq_vids, sq_perturb = evaluate_on_adversarial(
            model, sq_attack, square_loader, device, lpips_fn
        )
        row = _attack_row(
            name, "Square", adv_logits, sq_cp, sq_labels, sq_vids, sq_perturb
        )
        bb_rows.append(row)
        print(
            f"ACC={row['ACC']:.4f}  AUC={row['AUC']:.4f}  "
            f"ASR={row['ASR']:.4f}  PSNR={row['PSNR']:.1f}"
        )

    print_table(bb_rows, "Black-box: Square Attack (ε=8/255)")

    print("\nBuilding transfer adversarial examples (PGD-50) ...")
    transfer_asr: dict[str, dict[str, float]] = defaultdict(dict)
    transfer_v_auc: dict[str, dict[str, float]] = defaultdict(dict)

    for src_name, src_model in models.items():
        print(f"  Generating from {src_name} ...", end=" ", flush=True)
        attack = make_attack(
            torchattacks.PGD(src_model, eps=8 / 255, alpha=2 / 255, steps=50)
        )

        adv_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        for images, labels, _ in loader:
            images = images.to(device)
            labels_d = labels.to(device)
            adv_images = images.clone()
            fake_mask = labels_d == 1
            if fake_mask.any():
                adv_images[fake_mask] = attack(images[fake_mask], labels_d[fake_mask])
            adv_batches.append(adv_images.cpu())
            label_batches.append(labels)
        all_labels = torch.cat(label_batches).numpy()
        print("done")

        for tgt_name, tgt_model in models.items():
            adv_logits_list = []
            for adv_batch in adv_batches:
                with torch.no_grad():
                    adv_logits_list.append(tgt_model(adv_batch.to(device)).cpu())
            adv_logits_all = torch.cat(adv_logits_list)
            tgt_clean_preds = baseline_results[tgt_name]["preds"]
            tgt_vid_ids = baseline_results[tgt_name]["video_ids"]
            asr = attack_success_rate(
                tgt_clean_preds, adv_logits_all.argmax(1).numpy(), all_labels
            )
            v_auc = video_level_auc(
                torch.softmax(adv_logits_all, 1)[:, 1].numpy(), all_labels, tgt_vid_ids
            )
            transfer_asr[src_name][tgt_name] = asr
            transfer_v_auc[src_name][tgt_name] = v_auc

    model_names = list(models.keys())

    def _print_matrix(matrix: dict[str, dict[str, float]], title: str):
        print(f'\n{"─" * 70}')
        print(f"  {title}  (rows=source, cols=target)")
        print(f'{"─" * 70}')
        print("         " + "  ".join(f"{n:>10}" for n in model_names))
        for src in model_names:
            vals = "  ".join(f"{matrix[src][tgt]:>10.4f}" for tgt in model_names)
            print(f"{src:<9}{vals}")

    _print_matrix(transfer_asr, "Transfer ASR matrix")
    _print_matrix(transfer_v_auc, "Transfer video-AUC matrix")

    out_dir = Path(args.output_dir)

    def write_csv(rows, filename):
        if not rows:
            return
        path = out_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {path}")

    write_csv(baseline_rows, "baseline.csv")
    write_csv(wb_rows, "whitebox.csv")
    write_csv(bb_rows, "blackbox_square.csv")

    for matrix, filename in [
        (transfer_asr, "transfer_asr.csv"),
        (transfer_v_auc, "transfer_video_auc.csv"),
    ]:
        path = out_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source"] + model_names)
            for src in model_names:
                writer.writerow(
                    [src] + [f"{matrix[src][tgt]:.4f}" for tgt in model_names]
                )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
