"""
Celeb-DF v2 dataset loader.

Reads pre-extracted face frames from:
  {data_dir}/{category}/frames/{video_stem}/{frame}.png

Test split is defined by List_of_testing_videos.txt:
  {label} {category}/{video}.mp4
  e.g.: 1 Celeb-real/id0_0000.mp4
        0 Celeb-synthesis/id1_id0_0007.mp4
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)


def _normalize(img: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return torch.from_numpy(x).permute(2, 0, 1).float()


class CelebDFv2Dataset(Dataset):
    def __init__(self, data_dir: str, split_file: str | None = None):
        data_dir = Path(data_dir)
        if split_file is None:
            split_file = data_dir / "List_of_testing_videos.txt"

        self.samples: list[tuple[Path, int, str]] = []

        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # List_of_testing_videos.txt: 1=real, 0=fake
                # Model convention (deepfakebench): 0=real, 1=fake
                label = 1 - int(parts[0])
                video_rel = parts[1]  # e.g. "Celeb-real/id0_0000.mp4"
                video_path = Path(video_rel)
                category = video_path.parent.name  # "Celeb-real"
                video_stem = video_path.stem  # "id0_0000"
                video_id = f"{category}/{video_stem}"

                frames_dir = data_dir / category / "frames" / video_stem
                if not frames_dir.is_dir():
                    continue

                frames = sorted(frames_dir.glob("*.png"), key=lambda p: int(p.stem))
                for frame_path in frames:
                    self.samples.append((frame_path, label, video_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frame_path, label, video_id = self.samples[idx]
        img = cv2.imread(str(frame_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {frame_path}")
        tensor = _normalize(img)
        return tensor, label, video_id
