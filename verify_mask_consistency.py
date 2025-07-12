#!/usr/bin/env python3
"""
Verify mask consistency and sparsity patterns for pruned models.

This script checks:
1. Mask-weight consistency: every zero in weights matches zero in mask
2. Expected sparsity levels for N:M pruned layers
3. Layer eligibility for different pruning types

Usage:
    python verify_mask_consistency.py --ckpt model.pth --mask mask.pth [--nm-target 0.75]
"""

import argparse
import torch
from torchvision.models import resnet50
from trueobs import TrueOBS


def parse_args():
    parser = argparse.ArgumentParser(description="Verify mask consistency and pruning patterns")
    parser.add_argument("--ckpt", required=True, help="Path to pruned model checkpoint")
    parser.add_argument("--mask", required=True, help="Path to exported mask file")
    parser.add_argument("--nm-target", type=float, default=0.75, help="Expected sparsity for N:M pruning (default: 0.75 for 3:4)")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Sparsity tolerance (default: 0.01)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model and mask
    print(f"Loading checkpoint: {args.ckpt}")
    print(f"Loading mask: {args.mask}")
    
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    mask_dict = torch.load(args.mask, map_location="cpu")["mask"]
    
    print("\n" + "="*60)
    print("MASK CONSISTENCY CHECK")
    print("="*60)
    
    # 1. Mask consistency check (all layers)
    mismatches = []
    for name, p in model.named_parameters():
        if name.endswith(".weight"):
            if not torch.equal((p == 0), (mask_dict[name] == 0)):
                mismatches.append(name)
    
    if mismatches:
        print(f"❌ Mask mismatches found in {len(mismatches)} layers:")
        for name in mismatches:
            print(f"   - {name}")
        return False
    else:
        print("✅ Mask matches weights perfectly")
    
    print("\n" + "="*60)
    print("SPARSITY ANALYSIS")
    print("="*60)
    
    # 2. Analyze sparsity patterns by layer type
    nm_eligible = []
    nm_skipped = []
    sparsity_issues = []
    
    prunem = 4  # Assuming 3:4 pruning
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and len(module.weight.shape) > 1:
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                # Check eligibility for N:M pruning
                trueobs_layer = TrueOBS(module)
                weight = module.weight
                sparsity = (weight == 0).float().mean().item()
                
                if trueobs_layer.columns % prunem == 0:
                    nm_eligible.append((name, sparsity))
                    # Check if sparsity is close to target
                    if abs(sparsity - args.nm_target) > args.tolerance:
                        sparsity_issues.append((name, sparsity, args.nm_target))
                else:
                    nm_skipped.append((name, sparsity, trueobs_layer.columns))
    
    print(f"N:M Eligible Layers ({len(nm_eligible)}):")
    for name, sparsity in nm_eligible[:5]:  # Show first 5
        status = "✅" if abs(sparsity - args.nm_target) <= args.tolerance else "⚠️"
        print(f"  {status} {name}: {sparsity:.1%} sparse")
    if len(nm_eligible) > 5:
        print(f"  ... and {len(nm_eligible) - 5} more")
    
    print(f"\nSkipped Layers ({len(nm_skipped)}):")
    for name, sparsity, cols in nm_skipped:
        print(f"  ❌ {name}: {sparsity:.1%} sparse ({cols} cols, not divisible by {prunem})")
    
    if sparsity_issues:
        print(f"\n⚠️  Sparsity Issues ({len(sparsity_issues)}):")
        for name, actual, expected in sparsity_issues:
            print(f"   {name}: {actual:.1%} (expected {expected:.1%})")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success = len(mismatches) == 0 and len(sparsity_issues) == 0
    
    if success:
        print("✅ All checks passed!")
        print(f"   - Mask consistency: Perfect")
        print(f"   - N:M eligible layers: {len(nm_eligible)}")
        print(f"   - Average sparsity: {sum(s for _, s in nm_eligible) / len(nm_eligible):.1%}")
        print(f"   - Skipped layers: {len(nm_skipped)}")
    else:
        print("❌ Issues found:")
        if mismatches:
            print(f"   - Mask mismatches: {len(mismatches)}")
        if sparsity_issues:
            print(f"   - Sparsity issues: {len(sparsity_issues)}")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 