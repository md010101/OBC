#!/usr/bin/env python3
"""verify_sparse_accuracy.py

Quick utility to check overall weight sparsity and ImageNet validation
accuracy of a ResNet-50 checkpoint produced by the OBC pipeline.

Example
-------
python verify_sparse_accuracy.py \
       --ckpt rn50_14_75sparse.pth \
       --datapath /path/to/imagenet
"""

import argparse
import torch
from torchvision.models import resnet50

# Local helpers from the original OBC repository
from datautils import get_loaders
from modelutils import test


def parse_args():
    parser = argparse.ArgumentParser(description="Verify sparsity and accuracy of a pruned ResNet-50.")
    parser.add_argument("--ckpt", required=True, help="Path to the .pth checkpoint file to evaluate")
    parser.add_argument(
        "--datapath",
        required=True,
        help="Root directory of ImageNet containing 'train' and 'val' subfolders",
    )
    parser.add_argument("--batch", type=int, default=256, help="Validation batch size")
    parser.add_argument("--workers", type=int, default=16, help="Data-loader worker processes")
    return parser.parse_args()


def compute_sparsity(state_dict):
    zeros = 0
    total = 0
    for name, w in state_dict.items():
        if name.endswith(".weight") and w.dtype.is_floating_point:
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def main():
    args = parse_args()

    # Load checkpoint on CPU first (memory friendly) then move to GPU if available.
    state_dict = torch.load(args.ckpt, map_location="cpu")

    sparsity = compute_sparsity(state_dict)
    print(f"Total Sparsity: {sparsity:.2%}")

    # Build a vanilla ResNet-50 & load weights
    model = resnet50(weights=None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print("[Warning] Unexpected keys in state_dict:", unexpected)
    if missing:
        print("[Warning] Missing keys from state_dict:", missing)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Build data loaders (we only need the validation loader)
    _, val_loader = get_loaders(  # type: ignore
        "imagenet",
        path=args.datapath,
        batchsize=args.batch,
        workers=args.workers,
        noaug=True,
    )

    if val_loader is None:
        raise RuntimeError("Validation loader could not be created. Check the ImageNet path and structure.")

    # Evaluate top-1 accuracy (top-1)
    correct = 0
    total = 0
    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    print("Evaluating ...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.argmax(model(images), 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f"Model Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main() 