# VideoMAE Task Completion Detector — Plan

## Problem Statement

Given video recordings of a robot performing a placement task, classify each fixed-duration video clip into one of three states:

| Label | Class   | Meaning                                 |
|-------|---------|------------------------------------------|
| 0     | Doing   | Task is in progress (not yet complete)  |
| 1     | Failure | Task ended in failure                   |
| 2     | Success | Task ended in success                   |

During each episode, a robot performs the task. The episode always starts with label 0 (Doing) and eventually transitions to label 1 or 2 once the task outcome is determined.

---

## Dataset Format (LeRobot v2.1)

```
lerobot_vrh31/
  lerobot_vrh31_place_success_fail/       # main dataset (46 episodes)
  lerobot_vrh31_place_success_fail_2/     # (5 episodes)
  lerobot_vrh31_place_success_fail_3/     # (22 episodes)
  lerobot_vrh31_place_success_fail_4/     # (2 episodes)
  lerobot_vrh31_pick_success/             # (24 episodes)
    meta/
      info.json         # dataset metadata, fps=25, video codec=av1
      episodes.jsonl    # per-episode length
      tasks.jsonl       # task description string
    data/chunk-000/
      episode_XXXXXX.parquet   # per-frame data: timestamp, frame_index, observation.tasks.label
    videos/chunk-000/
      observation.images.color.head/episode_XXXXXX.mp4
      observation.images.color.head_right/episode_XXXXXX.mp4
      observation.images.color.left/episode_XXXXXX.mp4
      observation.images.color.right/episode_XXXXXX.mp4
```

**Key facts:**
- FPS: 25, codec: AV1
- Per-frame label in parquet (`observation.tasks.label`): 0, 1, or 2
- Episodes always start at label 0, transition to 1 or 2 at some frame
- Multiple sub-datasets are merged at training time

---

## Model Architecture

**Base model:** `MCG-NJU/videomae-base` (pre-trained on Kinetics-400)

- Fine-tune the VideoMAE transformer with a linear classification head replacing the original 400-class head
- Input: 16 uniformly sampled frames from a clip window
- Output: 3-class softmax (Doing / Failure / Success)

**Why VideoMAE:**
- Self-supervised pre-training on video makes it robust with limited labeled data
- Temporal masked autoencoding captures motion patterns critical for task state detection
- HuggingFace transformers integration simplifies fine-tuning

---

## Clip Extraction Strategy

### Training
For each labeled segment in an episode:
1. Find contiguous frame ranges for each label (e.g., frames 0–198 = Doing, frames 199–459 = Failure)
2. Slide a window of `CLIP_FRAMES` (default: 50 frames = 2 seconds) with stride `CLIP_STRIDE` (default: 25 frames = 1 second)
3. Sample 16 frames uniformly within each window → VideoMAE input
4. Label the entire window with the dominant label in that window

**Class imbalance handling:**
- Doing segments are long; terminal segments (Success/Failure) are short
- Apply weighted random sampling during training to balance classes

### Inference
Given a video file path + time window (start_sec, end_sec), or just a duration from the end:
1. Decode the specified frames from the mp4
2. Uniformly sample 16 frames
3. Run through VideoMAE → predict label

---

## Camera Selection

Default: `observation.images.color.head` (960×600, head-mounted front view)
- Configurable via `--camera` flag
- Can be extended to multi-camera fusion (not in scope for v1)

---

## File Structure

```
videomae/
  config.py          # all hyperparameters and paths
  dataset.py         # LeRobotClipDataset: reads parquet + decodes video clips
  model.py           # VideoMAEClassifier: loads HF model, swaps classification head
  train.py           # training loop with validation and checkpoint saving
  inference.py       # CLI: given video path + time range → predicted label
  requirements.txt
  Plan.md
```

---

## Training Pipeline

1. Load all episodes from all sub-datasets
2. Parse parquet for label segments per episode
3. Extract clips with sliding window → `(frames_tensor, label)` pairs
4. Split: 80% train / 20% val (stratified by label)
5. Fine-tune with:
   - AdamW optimizer, lr=1e-4, weight decay=0.01
   - Cosine LR schedule with linear warmup
   - Weighted cross-entropy loss (inverse class frequency)
   - Mixed precision (fp16)
6. Save best checkpoint by validation accuracy

---

## Inference

```bash
python inference.py \
  --video /path/to/episode.mp4 \
  --start 10.5 \
  --end 12.5 \
  --checkpoint checkpoints/best_model.pth
```

Or from the end of video:
```bash
python inference.py \
  --video /path/to/episode.mp4 \
  --last 2.0 \
  --checkpoint checkpoints/best_model.pth
```

Output: `Predicted: Success (confidence: 0.94)`

---

## Label Mapping

```python
LABEL_NAMES = {0: "Doing", 1: "Failure", 2: "Success"}
```



python train.py \
  --dataset_root /mnt/data/sftp/data/locht1/vr_data/lerobot_vrh31_classification \
  --epochs 20 \
  --batch_size 8 \
  --num_workers 8 \
  --wandb_project videomae-task-detector \
  --checkpoint_dir /mnt/data/sftp/data/locht1/vr_checkpoints/videomae_classification_imghead_2second
