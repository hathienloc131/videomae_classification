"""
Inference script for the VideoMAE task-completion classifier.

Given a video file and a time window, predicts whether the robot is:
  0 = Doing   (task in progress)
  1 = Failure (task failed)
  2 = Success (task succeeded)

Usage examples:
    # Explicit time window
    python inference.py --video episode.mp4 --start 10.5 --end 12.5 --checkpoint checkpoints/best_model.pth

    # Last N seconds of the video
    python inference.py --video episode.mp4 --last 2.0 --checkpoint checkpoints/best_model.pth

    # Use a specific camera stream from a LeRobot episode directory
    python inference.py \
        --dataset_root /Volumes/LocHT/lerobot_vrh31 \
        --subdir lerobot_vrh31_place_success_fail \
        --episode 0 \
        --start 10.0 --end 12.0 \
        --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import LABEL_NAMES, TrainConfig
from dataset import _decode_frames_decord, _uniform_sample
from model import load_checkpoint


def get_video_duration_and_fps(video_path: str):
    """Return (duration_seconds, fps) using decord or av."""
    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        duration = len(vr) / fps
        total_frames = len(vr)
        return duration, fps, total_frames
    except Exception:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        total_frames = stream.frames
        duration = total_frames / fps if fps > 0 else 0
        container.close()
        return duration, fps, total_frames


def predict(
    video_path: str,
    checkpoint_path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    last_sec: Optional[float] = None,
    num_frames: int = 16,
    image_size: int = 224,
    model_name: str = "MCG-NJU/videomae-base",
    device: Optional[str] = None,
) -> dict:
    """
    Run inference on a video clip.

    Args:
        video_path: path to the mp4 file
        checkpoint_path: path to .pth fine-tuned checkpoint
        start_sec: clip start time in seconds (used with end_sec)
        end_sec: clip end time in seconds
        last_sec: if set, use the last N seconds of the video
        num_frames: number of frames to sample for VideoMAE
        image_size: spatial size (default 224)
        model_name: HuggingFace model id for architecture
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        dict with keys: label (int), class_name (str), confidence (float), probabilities (dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    duration, fps, total_frames = get_video_duration_and_fps(video_path)

    # Resolve time window
    if last_sec is not None:
        end_sec = duration
        start_sec = max(0.0, duration - last_sec)
    elif start_sec is None or end_sec is None:
        raise ValueError("Provide either (--start and --end) or --last")

    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames)

    if end_frame <= start_frame:
        raise ValueError(f"Invalid time window: start={start_sec}s end={end_sec}s (fps={fps})")

    # Ensure we have enough frames
    if end_frame - start_frame < num_frames:
        # Expand window symmetrically
        deficit = num_frames - (end_frame - start_frame)
        start_frame = max(0, start_frame - deficit // 2)
        end_frame = min(total_frames, start_frame + num_frames)

    frame_indices = _uniform_sample(start_frame, end_frame, num_frames)

    # Decode & preprocess
    frames = _decode_frames_decord(video_path, frame_indices, image_size)  # T C H W
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    frames = torch.stack([normalize(frames[t]) for t in range(frames.shape[0])])  # T C H W
    pixel_values = frames.unsqueeze(0).to(device)  # 1 T C H W

    # Load model & run
    model = load_checkpoint(checkpoint_path, model_name=model_name).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = F.softmax(outputs.logits, dim=-1)[0]  # (3,)

    pred_label = int(probs.argmax().item())
    confidence = float(probs[pred_label].item())
    probabilities = {LABEL_NAMES[i]: float(probs[i].item()) for i in range(len(LABEL_NAMES))}

    return {
        "label": pred_label,
        "class_name": LABEL_NAMES[pred_label],
        "confidence": confidence,
        "probabilities": probabilities,
        "time_window": (start_sec, end_sec),
    }


def resolve_lerobot_video(
    dataset_root: str,
    subdir: str,
    episode: int,
    camera: str = "observation.images.color.head",
) -> str:
    """Resolve the mp4 path for a given LeRobot episode."""
    subdir_path = Path(dataset_root) / subdir
    info_path = subdir_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    chunk = episode // info["chunks_size"]
    video_rel = info["video_path"].format(
        episode_chunk=chunk,
        video_key=camera,
        episode_index=episode,
    )
    return str(subdir_path / video_rel)


def main():
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="VideoMAE task-completion inference")
    # Video source — either direct path or LeRobot episode
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", help="Direct path to an mp4 file")
    grp.add_argument("--dataset_root", help="LeRobot dataset root (use with --subdir and --episode)")

    p.add_argument("--subdir", default="lerobot_vrh31_place_success_fail",
                   help="Sub-dataset folder name (used with --dataset_root)")
    p.add_argument("--episode", type=int, default=0,
                   help="Episode index (used with --dataset_root)")
    p.add_argument("--camera", default=cfg.camera,
                   help="Camera stream name")

    # Time window
    time_grp = p.add_mutually_exclusive_group(required=True)
    time_grp.add_argument("--last", type=float,
                           help="Use the last N seconds of the video")
    time_grp.add_argument("--start", type=float,
                           help="Clip start time in seconds (use with --end)")

    p.add_argument("--end", type=float, help="Clip end time in seconds")

    p.add_argument("--checkpoint", required=True, help="Path to fine-tuned .pth checkpoint")
    p.add_argument("--model_name", default=cfg.model_name)
    p.add_argument("--num_frames", type=int, default=cfg.num_frames)
    p.add_argument("--image_size", type=int, default=cfg.image_size)
    p.add_argument("--device", default=None)

    args = p.parse_args()

    # Resolve video path
    if args.video:
        video_path = args.video
    else:
        video_path = resolve_lerobot_video(
            args.dataset_root, args.subdir, args.episode, args.camera
        )
        print(f"Resolved video: {video_path}")

    # Validate time args
    if args.start is not None and args.end is None:
        p.error("--start requires --end")

    result = predict(
        video_path=video_path,
        checkpoint_path=args.checkpoint,
        start_sec=args.start,
        end_sec=args.end,
        last_sec=args.last,
        num_frames=args.num_frames,
        image_size=args.image_size,
        model_name=args.model_name,
        device=args.device,
    )

    print(f"\nPredicted: {result['class_name']} (confidence: {result['confidence']:.4f})")
    print(f"Time window: {result['time_window'][0]:.2f}s → {result['time_window'][1]:.2f}s")
    print("Class probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 40)
        print(f"  {cls:10s}: {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
