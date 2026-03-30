"""
Adversarial robustness benchmark for deepfake detectors on Celeb-DF v2.

Attacks:
  White-box : FGSM (ε=2,4,8/255)  |  PGD-50 (ε=8/255)  |  C&W L2
  Black-box  : Square Attack (ε=8/255)
  Transfer   : PGD-50 adversarial examples from each source model
               evaluated on all target models → 4x4 ASR matrix

Usage:
  python main.py \\
    --checkpoint-dir /path/to/pretrained \\
    --data-dir       data/Celeb-DF-v2 \\
    --device         auto \\
    --models         xception effnetb4 ucf recce \\
    --batch-size     16 \\
    --cw-steps       200 \\
    --cw-samples     1000 \\   # cap C&W at 1000 frames (expensive)
    --square-samples 1000      # cap Square Attack at 1000 frames (very expensive)
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torchattacks

from dataset import CelebDFv2Dataset
from models import MODELS


def get_device(override: str | None = None) -> torch.device:
    if override and override != "auto":
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


CHECKPOINT_NAMES = {
    "xception": "xception_best.pth",
    "effnetb4": "effnb4_best.pth",
    "ucf": "ucf_best.pth",
    "recce": "recce_best.pth",
}


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


def attack_success_rate(clean_preds: np.ndarray, adv_preds: np.ndarray) -> float:
    return (clean_preds != adv_preds).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    all_logits, all_labels = [], []
    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate_on_adversarial(
    model: nn.Module,
    attack,
    loader: DataLoader,
    device: torch.device,
):
    adv_logits_list, clean_preds_list, labels_list = [], [], []
    model.eval()
    for images, labels, _ in loader:
        images = images.to(device)
        labels_d = labels.to(device)
        adv_images = attack(images, labels_d)
        with torch.no_grad():
            logits_clean = model(images)
            logits_adv = model(adv_images)
        clean_preds_list.append(logits_clean.argmax(1).cpu())
        adv_logits_list.append(logits_adv.cpu())
        labels_list.append(labels)
    return (
        torch.cat(adv_logits_list),
        torch.cat(clean_preds_list),
        torch.cat(labels_list),
    )


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

    baseline_rows = []
    baseline_results: dict[str, dict] = {}
    for name, model in models.items():
        logits, labels = evaluate(model, loader, device)
        acc, auc = compute_metrics(logits, labels)
        baseline_results[name] = {
            "logits": logits,
            "labels": labels,
            "preds": logits.argmax(1).numpy(),
        }
        baseline_rows.append({"model": name, "ACC": acc, "AUC": auc})
    print_table(baseline_rows, "Baseline (clean)")

    cw_loader = make_loader(args.cw_samples)
    if args.cw_samples is not None and args.cw_samples < len(dataset):
        print(f"C&W capped at {args.cw_samples} frames")

    wb_rows = []
    for name, model in models.items():
        labels = baseline_results[name]["labels"]
        clean_preds = baseline_results[name]["preds"]

        cheap_attacks = [
            ("FGSM-2", torchattacks.FGSM(model, eps=2 / 255)),
            ("FGSM-4", torchattacks.FGSM(model, eps=4 / 255)),
            ("FGSM-8", torchattacks.FGSM(model, eps=8 / 255)),
            ("PGD-50", torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=50)),
        ]

        for atk_name, attack in cheap_attacks:
            print(f"  {name} / {atk_name} ...", end=" ", flush=True)
            adv_logits, cp, _ = evaluate_on_adversarial(model, attack, loader, device)
            acc, auc = compute_metrics(adv_logits, labels)
            asr = attack_success_rate(cp.numpy(), adv_logits.argmax(1).numpy())
            wb_rows.append({"model": name, "attack": atk_name, "ACC": acc, "AUC": auc, "ASR": asr})
            print(f"ACC={acc:.4f}  AUC={auc:.4f}  ASR={asr:.4f}")

        print(f"  {name} / CW-L2 ...", end=" ", flush=True)
        cw_attack = torchattacks.CW(model, c=1.0, steps=args.cw_steps)
        cw_logits, _, cw_labels = evaluate_on_adversarial(model, cw_attack, cw_loader, device)
        cw_clean_preds = clean_preds[: len(cw_labels)]
        acc, auc = compute_metrics(cw_logits, cw_labels)
        asr = attack_success_rate(cw_clean_preds, cw_logits.argmax(1).numpy())
        wb_rows.append({"model": name, "attack": "CW-L2", "ACC": acc, "AUC": auc, "ASR": asr})
        print(f"ACC={acc:.4f}  AUC={auc:.4f}  ASR={asr:.4f}")

    print_table(wb_rows, "White-box attacks")

    square_loader = make_loader(args.square_samples)
    if args.square_samples is not None and args.square_samples < len(dataset):
        print(f"Square Attack capped at {args.square_samples} frames")

    bb_rows = []
    for name, model in models.items():
        clean_preds = baseline_results[name]["preds"]
        print(f"  {name} / Square ...", end=" ", flush=True)
        attack = torchattacks.Square(
            model, norm="Linf", eps=8 / 255, n_queries=5000, p_init=0.05
        )
        adv_logits, _, sq_labels = evaluate_on_adversarial(model, attack, square_loader, device)
        sq_clean_preds = clean_preds[: len(sq_labels)]
        acc, auc = compute_metrics(adv_logits, sq_labels)
        asr = attack_success_rate(sq_clean_preds, adv_logits.argmax(1).numpy())
        bb_rows.append({"model": name, "ACC": acc, "AUC": auc, "ASR": asr})
        print(f"ACC={acc:.4f}  AUC={auc:.4f}  ASR={asr:.4f}")

    print_table(bb_rows, "Black-box: Square Attack (ε=8/255)")

    print("\nBuilding transfer adversarial examples (PGD-50) ...")
    transfer_asr: dict[str, dict[str, float]] = defaultdict(dict)

    for src_name, src_model in models.items():
        print(f"  Generating from {src_name} ...", end=" ", flush=True)
        attack = torchattacks.PGD(src_model, eps=8 / 255, alpha=2 / 255, steps=50)

        adv_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        for images, labels, _ in loader:
            images = images.to(device)
            labels_d = labels.to(device)
            adv_images = attack(images, labels_d)
            adv_batches.append(adv_images.cpu())
            label_batches.append(labels)
        print("done")

        for tgt_name, tgt_model in models.items():
            adv_logits_list = []
            for adv_batch in adv_batches:
                with torch.no_grad():
                    logits_adv = tgt_model(adv_batch.to(device))
                adv_logits_list.append(logits_adv.cpu())
            adv_logits_all = torch.cat(adv_logits_list)
            tgt_clean_preds = baseline_results[tgt_name]["preds"]
            asr = attack_success_rate(tgt_clean_preds, adv_logits_all.argmax(1).numpy())
            transfer_asr[src_name][tgt_name] = asr

    model_names = list(models.keys())
    print(f'\n{"─" * 70}')
    print("  Transfer ASR matrix  (rows=source, cols=target)")
    print(f'{"─" * 70}')
    header = "         " + "  ".join(f"{n:>10}" for n in model_names)
    print(header)
    for src in model_names:
        row_vals = "  ".join(f"{transfer_asr[src][tgt]:>10.4f}" for tgt in model_names)
        print(f"{src:<9}{row_vals}")

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

    transfer_path = out_dir / "transfer_matrix.csv"
    with open(transfer_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source"] + model_names)
        for src in model_names:
            writer.writerow(
                [src] + [f"{transfer_asr[src][tgt]:.4f}" for tgt in model_names]
            )
    print(f"Saved {transfer_path}")


if __name__ == "__main__":
    main()
