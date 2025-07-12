import os
from typing import Dict, Union, Any, Optional

import torch
import torch.nn as nn


def _tensor_to_mask(t: torch.Tensor) -> torch.Tensor:
    """Convert weight tensor to binary float mask (1.0 keep, 0.0 prune)."""
    # Ensure dtype float32 for SNOWS compatibility
    return (t != 0).float()


def _is_batchnorm_layer(module_name: str) -> bool:
    """Check if a module name corresponds to a BatchNorm layer."""
    bn_patterns = [
        'bn1', 'bn2', 'bn3',  # Direct BN layers
        'downsample.1',        # Downsample BN layers in ResNet
        '.bn',                 # Any layer containing .bn
        'norm'                 # LayerNorm or other norm layers
    ]
    return any(pattern in module_name for pattern in bn_patterns)


def _is_conv_layer(model: nn.Module, module_name: str) -> bool:
    """Check if a module name corresponds to a Conv2d layer."""
    try:
        # Navigate to the module using the name
        module = model
        for part in module_name.split('.'):
            module = getattr(module, part)
        return isinstance(module, nn.Conv2d)
    except AttributeError:
        return False


def generate_mask_dict_from_model(model: nn.Module, *, device: Union[str, torch.device] = "cpu") -> Dict[str, torch.Tensor]:
    """Return a per-layer mask dictionary derived from an in-memory PyTorch model.

    Only entries whose parameter name ends with ``.weight`` are considered, as SNOWS
    applies masks exclusively to weight tensors.

    Parameters
    ----------
    model : nn.Module
        The pruned model instance.
    device : str or torch.device, optional
        Device on which the returned masks will reside.  Defaults to CPU so that
        mask files remain backend-agnostic and lightweight.

    Returns
    -------
    Dict[str, torch.Tensor]
        Mapping from parameter names (e.g. ``'layer1.0.conv1.weight'``) to binary
        float32 masks of identical shape.
    """
    mask_dict: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name.endswith(".weight"):
            mask = _tensor_to_mask(param.data).to(device)
            mask_dict[name] = mask
    return mask_dict


def generate_mask_dict_from_state_dict(state_dict: Dict[str, torch.Tensor], *, device: Union[str, torch.device] = "cpu") -> Dict[str, torch.Tensor]:
    """Return a mask dictionary from a state_dict already on CPU/GPU.

    Only ``*.weight`` entries are processed; biases and other buffers are
    ignored.
    """
    mask_dict: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.endswith(".weight"):
            mask_dict[name] = _tensor_to_mask(tensor).to(device)
    return mask_dict


def generate_snows_mask_dict_from_model(model: nn.Module, *, device: Union[str, torch.device] = "cpu") -> Dict[str, torch.Tensor]:
    """Return a SNOWS-compatible mask dictionary from an in-memory PyTorch model.

    Only Conv2d layers are included. BatchNorm layers are filtered out.
    Keys are module names (without '.weight' suffix).

    Parameters
    ----------
    model : nn.Module
        The pruned model instance.
    device : str or torch.device, optional
        Device on which the returned masks will reside.  Defaults to CPU so that
        mask files remain backend-agnostic and lightweight.

    Returns
    -------
    Dict[str, torch.Tensor]
        Mapping from module names (e.g. ``'layer1.0.conv1'``) to binary
        float32 masks of identical shape.
    """
    mask_dict: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name.endswith(".weight"):
            module_name = name[:-7]  # Remove '.weight' suffix
            
            # Only include Conv2d layers, exclude BatchNorm
            if not _is_batchnorm_layer(module_name) and _is_conv_layer(model, module_name):
                mask = _tensor_to_mask(param.data).to(device)
                mask_dict[module_name] = mask
    return mask_dict


def generate_snows_mask_dict_from_state_dict(state_dict: Dict[str, torch.Tensor], model: Optional[nn.Module] = None, *, device: Union[str, torch.device] = "cpu") -> Dict[str, torch.Tensor]:
    """Return a SNOWS-compatible mask dictionary from a state_dict.

    Only Conv2d layers are included. BatchNorm layers are filtered out.
    Keys are module names (without '.weight' suffix).

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        State dict containing the weights to generate masks from.
    model : nn.Module, optional
        Model instance to check layer types. If None, uses heuristic filtering.
    device : str or torch.device, optional
        Device on which the returned masks will reside.

    Returns
    -------
    Dict[str, torch.Tensor]
        Mapping from module names (e.g. ``'layer1.0.conv1'``) to binary
        float32 masks of identical shape.
    """
    mask_dict: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.endswith(".weight"):
            module_name = name[:-7]  # Remove '.weight' suffix
            
            # Filter out BatchNorm layers
            if not _is_batchnorm_layer(module_name):
                # If model is provided, check if it's a Conv2d layer
                if model is None:
                    # No model provided, use heuristic filtering (already filtered BN)
                    mask_dict[module_name] = _tensor_to_mask(tensor).to(device)
                elif _is_conv_layer(model, module_name):
                    mask_dict[module_name] = _tensor_to_mask(tensor).to(device)
    return mask_dict


def save_snows_mask(mask_dict: Dict[str, torch.Tensor], target_path: str) -> None:
    """Save mask_dict in SNOWS-compatible ``.pth`` structure.

    The file will contain a single key ``'mask'`` whose value is the provided
    dictionary.  Intermediate directories are created automatically.
    """
    dirname = os.path.dirname(target_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save({"mask": mask_dict}, target_path)


def save_snows_mask_flat(mask_dict: Dict[str, torch.Tensor], target_path: str) -> None:
    """Save mask_dict in flat SNOWS-compatible format.

    The file will contain the mask dictionary directly without nesting.
    Intermediate directories are created automatically.

    Parameters
    ----------
    mask_dict : Dict[str, torch.Tensor]
        Dictionary mapping module names to mask tensors.
    target_path : str
        Path where the mask file will be saved.
    """
    dirname = os.path.dirname(target_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save(mask_dict, target_path) 