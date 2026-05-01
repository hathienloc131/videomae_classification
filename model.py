"""
VideoMAE-based 3-class classifier (Doing / Failure / Success).

Wraps HuggingFace VideoMAEForVideoClassification, which already includes
a linear classification head. We just configure num_labels=3 and load
pre-trained weights with the head randomly initialised.
"""

import torch
import torch.nn as nn
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

from config import LABEL_NAMES, NUM_CLASSES


def build_model(model_name: str = "MCG-NJU/videomae-base", freeze_backbone: bool = False) -> nn.Module:
    """
    Load VideoMAE with a fresh 3-class head.

    Args:
        model_name: HuggingFace model id
        freeze_backbone: if True, only train the classification head (useful for
                         very small datasets or quick experiments)
    """
    id2label = {str(k): v for k, v in LABEL_NAMES.items()}
    label2id = {v: str(k) for k, v in LABEL_NAMES.items()}

    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # head size changes from 400 → 3
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad_(False)

    return model


def load_checkpoint(checkpoint_path: str, model_name: str = "MCG-NJU/videomae-base") -> nn.Module:
    """Load a fine-tuned checkpoint for inference."""
    model = build_model(model_name)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    return model
