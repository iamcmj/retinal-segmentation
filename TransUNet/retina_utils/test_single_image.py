from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def test_single_image(
    model: torch.nn.Module,
    image_path: str,
    img_size: int = 512,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    resize_back: bool = True,
) -> np.ndarray:
    """
    Run 2D inference on a single retinal fundus image and optionally save the mask as PNG.

    Args:
        model: Segmentation model (already loaded with weights).
        image_path: Path to the input RGB fundus image.
        img_size: Resize target for the network input.
        device: Torch device. Defaults to CUDA if available.
        save_path: Optional path to save the predicted mask PNG.
        resize_back: If True, resize the mask back to the original image size before saving/returning.

    Returns:
        Predicted mask as a uint8 numpy array (0/255).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(path).convert("RGB")
    original_size = image.size
    transform = _build_transform(img_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
        mask_np = prediction.cpu().numpy().astype(np.uint8)

    mask_image = Image.fromarray(mask_np * 255, mode="L")
    if resize_back and mask_image.size != original_size:
        mask_image = mask_image.resize(original_size, resample=Image.NEAREST)
        mask_np = np.array(mask_image, dtype=np.uint8)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mask_image.save(save_path)

    return mask_np
