import torch
import torchvision.transforms as transforms
from typing import List, Tuple

from torchvision.models.video import r3d_18, R3D_18_Weights


def load_backbone(*, backbone: str, device: torch.device) -> Tuple[torch.nn.Module, transforms.Compose, List[str]]:
    """Load a video backbone with its preprocessing transforms and label list.

    Args:
        backbone: Backbone name, e.g. "r3d18" or "i3d" (case-insensitive).
        device: Target torch device.

    Returns:
        (model, transform, labels)
    """
    bb = (backbone or "r3d18").lower()

    if bb == "r3d18":
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights).eval().to(device)
        labels = list(weights.meta["categories"])  # type: ignore[index]
        video_transform = weights.transforms()
        transform = transforms.Compose([
            transforms.Resize(video_transform.resize_size, antialias=True),
            transforms.CenterCrop(video_transform.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=video_transform.mean, std=video_transform.std),
        ])
        return model, transform, labels

    if bb == "i3d":
        try:
            model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
            model = model.eval().to(device)
        except Exception as e:
            raise RuntimeError(
                "Failed to load I3D backbone via torch.hub. "
                "Ensure internet access and that 'facebookresearch/pytorchvideo' is reachable. "
                f"Original error: {e}"
            )
        transform = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ])
        labels = list(R3D_18_Weights.KINETICS400_V1.meta["categories"])  # type: ignore[index]
        return model, transform, labels

    raise ValueError(f"Unsupported backbone: {backbone}")


