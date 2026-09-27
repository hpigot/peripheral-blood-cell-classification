"""Model zoo (torchvision, ImageNet-pretrained) and shared preprocessing.

The preprocessing constants are the contract with the C++ edge code: resize
to INPUT_SIZE, RGB, scale to [0, 1], normalize with MEAN/STD, NCHW float32.
Change them here and in edge/cpp/main.cpp together.
"""

from __future__ import annotations

INPUT_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

ARCHS = ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18")


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    from torch import nn
    from torchvision import models

    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    weights = "DEFAULT" if pretrained else None
    m = getattr(models, arch)(weights=weights)
    if arch.startswith(("mobilenet", "efficientnet")):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:  # resnet
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_checkpoint(path, device):
    """The model (eval mode, on ``device``) and class names from a bloodcell-train checkpoint."""
    import torch

    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"]


# ColorJitter strengths. "strong" was fixed before scoring any retrained model on
# Raabin-WBC, wide enough to cover its nuclei (magenta where PBC's are blue-purple).
COLOUR_JITTER = {
    "standard": {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.03},
    "strong": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.4, "hue": 0.1},
}


def transforms(train: bool, colour: str = "standard"):
    from torchvision import transforms as T

    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if not train:
        return T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), *norm])
    # Cells are rotation- and flip-invariant; stain/illumination varies by lab.
    return T.Compose(
        [
            T.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            T.ColorJitter(**COLOUR_JITTER[colour]),
            *norm,
        ]
    )


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
