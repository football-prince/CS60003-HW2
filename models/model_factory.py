from __future__ import annotations

from typing import Any

from torch import nn
from torchvision import models

from models.resnet_attention import (
    attention_resnet18,
    attention_resnet34,
    remove_classifier_keys,
)


def _normalize_attention(attention: str | None) -> str | None:
    if attention is None:
        return None
    attention = attention.lower()
    return None if attention == "none" else attention


def _get_resnet_weights(model_name: str, pretrained: bool):
    if not pretrained:
        return None
    if model_name == "resnet18":
        return models.ResNet18_Weights.IMAGENET1K_V1
    if model_name == "resnet34":
        return models.ResNet34_Weights.IMAGENET1K_V1
    raise ValueError(f"Unsupported ResNet model: {model_name}")


def _build_plain_resnet(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    weights = _get_resnet_weights(model_name, pretrained)
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.pretrained_load_info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "missing_count": 0,
        "unexpected_count": 0,
    }
    return model


def _build_attention_resnet(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    attention: str,
) -> nn.Module:
    if model_name == "resnet18":
        model = attention_resnet18(num_classes=num_classes, attention=attention)
    elif model_name == "resnet34":
        model = attention_resnet34(num_classes=num_classes, attention=attention)
    else:
        raise ValueError(f"Unsupported attention ResNet model: {model_name}")

    load_info: dict[str, Any] = {
        "missing_keys": [],
        "unexpected_keys": [],
        "missing_count": 0,
        "unexpected_count": 0,
    }
    if pretrained:
        weights = _get_resnet_weights(model_name, pretrained=True)
        state_dict = remove_classifier_keys(weights.get_state_dict(progress=True))
        incompatible = model.load_state_dict(state_dict, strict=False)
        load_info = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "missing_count": len(incompatible.missing_keys),
            "unexpected_count": len(incompatible.unexpected_keys),
        }
        print(
            "Loaded torchvision ResNet weights with strict=False: "
            f"missing={load_info['missing_count']}, "
            f"unexpected={load_info['unexpected_count']}"
        )

    model.pretrained_load_info = load_info
    return model


def _build_timm_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Please install timm to use vit_tiny or swin_t.") from exc

    timm_name = {
        "vit_tiny": "vit_tiny_patch16_224",
        "swin_t": "swin_tiny_patch4_window7_224",
    }[model_name]
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    model.pretrained_load_info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "missing_count": 0,
        "unexpected_count": 0,
    }
    return model


def build_model(
    model_name: str,
    num_classes: int = 102,
    pretrained: bool = True,
    attention: str | None = None,
) -> nn.Module:
    """Build a Flowers102 classifier.

    Supported:
    - resnet18/resnet34 with optional attention=None, "se", or "cbam"
    - vit_tiny and swin_t from timm
    """
    model_name = model_name.lower()
    attention = _normalize_attention(attention)

    if model_name in {"resnet18", "resnet34"}:
        if attention is None:
            return _build_plain_resnet(model_name, num_classes, pretrained)
        if attention not in {"se", "cbam"}:
            raise ValueError("ResNet attention must be one of: none, se, cbam")
        return _build_attention_resnet(model_name, num_classes, pretrained, attention)

    if model_name in {"vit_tiny", "swin_t"}:
        if attention is not None:
            print(f"Ignoring attention={attention} for {model_name}.")
        return _build_timm_model(model_name, num_classes, pretrained)

    raise ValueError(f"Unsupported model: {model_name}")

