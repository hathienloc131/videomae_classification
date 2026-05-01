"""
LeRobot v2.1 dataset reader for VideoMAE task-completion classification.

Each episode contains per-frame labels (0=Doing, 1=Failure, 2=Success).
This module:
  1. Parses every episode's parquet to find contiguous label segments.
  2. Slides a fixed window over each segment to produce (video_path, start_frame, end_frame, label) records.
  3. Decodes the requested frame range from the AV1-encoded mp4 at __getitem__ time.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _decode_frames_decord(video_path: str, frame_indices: List[int], image_size: int) -> torch.Tensor:
    """Decode specific frames from an mp4 using decord. Returns (T, C, H, W) float32 in [0,1]."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frames = vr.get_batch(frame_indices)  # (T, H, W, C) uint8
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    except Exception:
        frames = _decode_frames_av(video_path, frame_indices)

    # Resize to image_size x image_size
    resize = transforms.Resize((image_size, image_size), antialias=True)
    frames = resize(frames.view(-1, *frames.shape[-2:]).unsqueeze(0)).squeeze(0)
    frames = resize(frames)
    return frames  # (T, C, H, W)


def _decode_frames_av(video_path: str, frame_indices: List[int]) -> torch.Tensor:
    """Fallback AV-based decoder when decord is unavailable."""
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    target_set = set(frame_indices)
    collected = {}
    for i, frame in enumerate(container.decode(stream)):
        if i in target_set:
            img = frame.to_image()
            arr = np.array(img, dtype=np.float32) / 255.0  # H W C
            collected[i] = torch.from_numpy(arr).permute(2, 0, 1)  # C H W
        if len(collected) == len(target_set):
            break
    container.close()
    return torch.stack([collected[i] for i in frame_indices], dim=0)  # T C H W


def _uniform_sample(start: int, end: int, n: int) -> List[int]:
    """Return n uniformly spaced frame indices in [start, end)."""
    indices = np.linspace(start, end - 1, n, dtype=int).tolist()
    return indices


def _find_label_segments(df: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """
    Returns list of (start_frame, end_frame_exclusive, label) for contiguous
    label runs in the parquet dataframe.
    """
    labels = df["observation.tasks.label"].values
    segments = []
    i = 0
    while i < len(labels):
        label = labels[i]
        j = i
        while j < len(labels) and labels[j] == label:
            j += 1
        segments.append((i, j, int(label)))
        i = j
    return segments


class LeRobotClipDataset(Dataset):
    """
    Sliding-window clip dataset built from all LeRobot v2.1 sub-datasets.

    Args:
        dataset_root: root directory containing sub-dataset folders
        dataset_subdirs: which sub-folders to include
        camera: which video stream to use
        clip_frames: window size in frames (e.g. 50 = 2 s at 25 fps)
        clip_stride: stride between windows
        num_frames: number of frames to sample per clip for the model
        image_size: spatial size to resize frames to
        split: "train" or "val"
        val_split: fraction of episodes reserved for validation
        seed: random seed for split
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_subdirs: List[str],
        camera: str = "observation.images.color.head",
        clip_frames: int = 50,
        clip_stride: int = 25,
        num_frames: int = 16,
        image_size: int = 224,
        split: str = "train",
        val_split: float = 0.2,
        seed: int = 42,
    ):
        self.camera = camera
        self.clip_frames = clip_frames
        self.clip_stride = clip_stride
        self.num_frames = num_frames
        self.image_size = image_size
        self.split = split

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Collect all (video_path, start_frame, end_frame_exclusive, label) records
        records = self._collect_records(dataset_root, dataset_subdirs)

        # Deterministic train/val split by episode-level grouping
        rng = np.random.default_rng(seed)
        all_indices = np.arange(len(records))
        rng.shuffle(all_indices)
        split_at = int(len(records) * (1 - val_split))
        if split == "train":
            chosen = all_indices[:split_at]
        else:
            chosen = all_indices[split_at:]

        self.records = [records[i] for i in chosen]

        # Build class weights for weighted sampling (inverse frequency)
        label_arr = np.array([r[3] for r in self.records])
        counts = np.bincount(label_arr, minlength=3).astype(float)
        counts = np.where(counts == 0, 1, counts)
        self.class_weights = (1.0 / counts)
        self.sample_weights = self.class_weights[label_arr]

    def _collect_records(
        self,
        dataset_root: str,
        dataset_subdirs: List[str],
    ) -> List[Tuple[str, int, int, int]]:
        records = []
        for subdir in dataset_subdirs:
            subdir_path = Path(dataset_root) / subdir
            if not subdir_path.exists():
                print(f"[dataset] Skipping missing sub-dataset: {subdir_path}")
                continue

            info_path = subdir_path / "meta" / "info.json"
            with open(info_path) as f:
                info = json.load(f)

            video_path_tpl = info["video_path"]  # e.g. videos/chunk-{:03d}/{video_key}/episode_{:06d}.mp4
            data_path_tpl = info["data_path"]
            chunks_size = info["chunks_size"]

            episodes_path = subdir_path / "meta" / "episodes.jsonl"
            episodes = []
            with open(episodes_path) as f:
                for line in f:
                    episodes.append(json.loads(line))

            for ep in episodes:
                ep_idx = ep["episode_index"]
                ep_len = ep["length"]
                chunk = ep_idx // chunks_size

                parquet_rel = data_path_tpl.format(
                    episode_chunk=chunk, episode_index=ep_idx
                )
                parquet_path = subdir_path / parquet_rel
                if not parquet_path.exists():
                    continue

                df = pd.read_parquet(parquet_path, columns=["observation.tasks.label"])
                segments = _find_label_segments(df)

                # Resolve video path for this camera
                # video_path template uses {video_key} as named placeholder
                video_rel = video_path_tpl.format(
                    episode_chunk=chunk,
                    video_key=self.camera,
                    episode_index=ep_idx,
                )
                video_path = str(subdir_path / video_rel)
                if not os.path.exists(video_path):
                    continue

                for seg_start, seg_end, label in segments:
                    seg_len = seg_end - seg_start
                    if seg_len < self.num_frames:
                        # Segment too short: keep as single clip
                        records.append((video_path, seg_start, seg_end, label))
                        continue
                    # Slide window
                    w_start = seg_start
                    while w_start + self.clip_frames <= seg_end:
                        records.append((video_path, w_start, w_start + self.clip_frames, label))
                        w_start += self.clip_stride
                    # Trailing clip (avoids missing last frames)
                    if w_start < seg_end and (seg_end - w_start) >= self.num_frames:
                        records.append((video_path, seg_end - self.clip_frames, seg_end, label))

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        video_path, start_frame, end_frame, label = self.records[idx]
        frame_indices = _uniform_sample(start_frame, end_frame, self.num_frames)

        frames = _decode_frames_decord(video_path, frame_indices, self.image_size)  # T C H W
        # Normalize each frame
        frames = torch.stack([self.normalize(frames[t]) for t in range(frames.shape[0])])  # T C H W
        # VideoMAE expects (C, T, H, W)
        pixel_values = frames.permute(1, 0, 2, 3)  # C T H W

        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}
