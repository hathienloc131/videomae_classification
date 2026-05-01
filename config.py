from dataclasses import dataclass, field
from typing import List


LABEL_NAMES = {0: "Doing", 1: "Failure", 2: "Success"}
NUM_CLASSES = 3

# Sub-dataset directories under the root
DATASET_SUBDIRS = [
    "lerobot_vrh31_place_success_fail",
    "lerobot_vrh31_place_success_fail_2",
    "lerobot_vrh31_place_success_fail_3",
    "lerobot_vrh31_place_success_fail_4",
    "lerobot_vrh31_pick_success",
]


@dataclass
class TrainConfig:
    # Paths
    dataset_root: str = "/mnt/data/sftp/data/locht1/vr_data/lerobot_vrh31_classification"
    dataset_subdirs: List[str] = field(default_factory=lambda: DATASET_SUBDIRS)
    camera: str = "observation.images.color.head"
    checkpoint_dir: str = "checkpoints"

    # Model
    model_name: str = "MCG-NJU/videomae-base"

    # Clip sampling
    num_frames: int = 16          # frames fed to VideoMAE
    clip_frames: int = 50         # sliding window size (2 s at 25 fps)
    clip_stride: int = 25         # stride between windows (1 s)

    # Image size expected by VideoMAE
    image_size: int = 224

    # Training
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 4
    fp16: bool = True

    # Logging
    log_every: int = 20
