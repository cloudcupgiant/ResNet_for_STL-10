"""Model factory functions and architecture-specific customization helpers."""

from __future__ import annotations

from typing import Any

from torch import nn

from .small_resnet import make_activation, small_resnet18


SUPPORTED_MODELS = ("small_resnet18", "shufflenetv2_x1_0", "efficientnet_b0")


# Only feature activations should be replaced; classifier heads and SE gates are handled separately.
REPLACEABLE_ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.SiLU)


def replace_activation_modules(module: nn.Module, activation: str) -> nn.Module:
    """Replace feature activations while keeping classifier and SE sigmoid gates intact."""
    for name, child in module.named_children():
        if isinstance(child, REPLACEABLE_ACTIVATIONS):
            setattr(module, name, make_activation(activation))
        else:
            replace_activation_modules(child, activation)
    return module


def replace_se_scale_activation(module: nn.Module, activation: str) -> nn.Module:
    """Replace EfficientNet Squeeze-and-Excitation scale gates only."""
    for child in module.modules():
        # Torchvision exposes SE gates by class name rather than a stable public type.
        if child.__class__.__name__ == "SqueezeExcitation" and hasattr(child, "scale_activation"):
            child.scale_activation = make_activation(activation)
    return module


def shufflenetv2_x1_0(num_classes: int = 10, dropout: float = 0.0, activation: str = "relu") -> nn.Module:
    """Build ShuffleNetV2 x1.0 from scratch for STL-10 classification."""
    from torchvision.models import shufflenet_v2_x1_0

    model = shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
    replace_activation_modules(model, activation)
    if dropout > 0:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return model


def efficientnet_b0(num_classes: int = 10, dropout: float = 0.0, se_activation: str = "sigmoid") -> nn.Module:
    """Build EfficientNet-B0 from scratch with native SiLU and configurable SE gates."""
    from torchvision.models import efficientnet_b0 as tv_efficientnet_b0

    model = tv_efficientnet_b0(weights=None, num_classes=num_classes, dropout=dropout)
    replace_se_scale_activation(model, se_activation)
    return model


def create_model(name: str, **kwargs: Any) -> nn.Module:
    """Construct a supported model from a shared keyword-argument dictionary."""
    if name == "small_resnet18":
        return small_resnet18(
            num_classes=kwargs["num_classes"],
            dropout=kwargs.get("dropout", 0.2),
            activation=kwargs.get("activation", "relu"),
            base_channels=kwargs.get("base_channels", 64),
        )
    if name == "shufflenetv2_x1_0":
        return shufflenetv2_x1_0(
            num_classes=kwargs["num_classes"],
            dropout=kwargs.get("dropout", 0.0),
            activation=kwargs.get("activation", "relu"),
        )
    if name == "efficientnet_b0":
        return efficientnet_b0(
            num_classes=kwargs["num_classes"],
            dropout=kwargs.get("dropout", 0.0),
            se_activation=kwargs.get("se_activation", "sigmoid"),
        )
    raise ValueError(f"Unsupported model: {name}")


def checkpoint_model_name(model_kwargs: dict[str, Any]) -> str:
    """Read the architecture name saved in a checkpoint, defaulting to the baseline."""
    return str(model_kwargs.get("model_name", "small_resnet18"))


def create_model_from_checkpoint_kwargs(model_kwargs: dict[str, Any]) -> nn.Module:
    """Rebuild a model with the same constructor kwargs stored in a checkpoint."""
    model_name = checkpoint_model_name(model_kwargs)
    return create_model(model_name, **model_kwargs)
