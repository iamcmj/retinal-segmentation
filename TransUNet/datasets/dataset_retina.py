import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXTENSIONS = {".ppm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}


class RetinaVesselDataset(Dataset):
    """
    Retina vessel segmentation dataset.

    The dataset automatically discovers image/mask pairs under the given root directory.
    Expected structure (consistent across raw and processed):
        dataset/DRIVE/training/images/*.tif        (fundus images)
        dataset/DRIVE/training/masks/*.gif         (vessel labels)

    Image and mask files are matched by shared stem with optional suffix stripping
    (e.g., "21_training.tif" <-> "21_training.gif" or "21_training_mask.gif").
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: Optional[str] = "training",
        image_size: int = 512,
        augment: bool = True,
        prefer_processed: bool = True,
        return_dict: bool = False,
    ) -> None:
        """
        Args:
            root_dir: Dataset root (e.g., dataset/DRIVE with training/images and training/masks).
            split: Sub-folder to use (e.g., "training" or "test"). If None, use root directly.
            image_size: Output square size fed into the network.
            augment: Apply random flips/rotation and brightness/contrast jittering.
            prefer_processed: Prefer dataset/processed/* if present before raw folders.
            return_dict: If True, return {"image": tensor, "label": tensor} instead of a tuple.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.prefer_processed = prefer_processed
        self.return_dict = return_dict
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

        self.images_dir, self.masks_dir = self._locate_image_and_mask_dirs()
        self.samples = self._gather_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self._apply_augmentations(image, mask)
        image = TF.resize(
            image, (self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask, (self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST
        )

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Binarize mask after resizing to keep crisp edges.
        mask_np = (np.array(mask, dtype=np.uint8) > 0).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_np)

        if self.return_dict:
            return {"image": image, "label": mask_tensor}
        return image, mask_tensor


    def _apply_augmentations(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if not self.augment:
            return image, mask

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.5:
            angle = random.uniform(-15.0, 15.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        if random.random() < 0.5:
            image = self.color_jitter(image)

        return image, mask

    def _locate_image_and_mask_dirs(self) -> Tuple[Path, Path]:
        """
        Find matching images/ and masks/ folders, preferring processed data when available.
        """
        candidates: List[Path] = []
        if self.split:
            if self.prefer_processed:
                candidates.append(self.root_dir / "processed" / self.split)
            candidates.append(self.root_dir / self.split)
        if self.prefer_processed:
            candidates.append(self.root_dir / "processed")
        candidates.append(self.root_dir)

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                unique_candidates.append(candidate)
                seen.add(candidate)

        for base in unique_candidates:
            images_dir = base / "images"
            masks_dir = base / "masks"
            if images_dir.is_dir() and masks_dir.is_dir():
                if self._dir_has_valid_images(images_dir) and self._dir_has_valid_images(masks_dir):
                    return images_dir, masks_dir

        raise FileNotFoundError(
            f"Could not locate images/ and masks/ folders under {self.root_dir}. "
            f"Tried: {', '.join(str(c) for c in unique_candidates)}"
        )

    def _gather_samples(self) -> List[Tuple[Path, Path]]:
        mask_index = self._build_mask_index()
        samples: List[Tuple[Path, Path]] = []
        for image_path in sorted(self.images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            base_key = image_path.stem.lower()
            mask_path = mask_index.get(base_key)
            if mask_path is not None:
                samples.append((image_path, mask_path))

        if not samples:
            raise RuntimeError(
                f"No image/mask pairs found using stems from {self.images_dir} and {self.masks_dir}"
            )
        return samples

    def _build_mask_index(self) -> dict:
        mask_index = {}
        for mask_path in self.masks_dir.iterdir():
            if not mask_path.is_file() or mask_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            stem = mask_path.stem.lower()
            candidate_keys = {stem}
            strip_suffixes = ["_mask", "_manual1", "_manual", "_1st_manual", "_manual_1"]
            for suffix in strip_suffixes:
                if stem.endswith(suffix):
                    candidate_keys.add(stem[: -len(suffix)])

            for key in candidate_keys:
                # First-come-first-serve so that 1st_manual (checked before mask) is preferred.
                mask_index.setdefault(key, mask_path)
        return mask_index

    def _dir_has_valid_images(self, directory: Path) -> bool:
        return any(
            path.is_file() and path.suffix.lower() in VALID_EXTENSIONS for path in directory.iterdir()
        )


def build_retina_dataset(
    root_dirs: Union[str, Path, List[Union[str, Path]]],
    split: Optional[str] = "training",
    image_size: int = 512,
    augment: bool = True,
    prefer_processed: bool = True,
    return_dict: bool = False,
):
    """
    Build a dataset from one or many root directories. When multiple roots are provided,
    their samples are concatenated into a single dataset without altering folder structure.
    """
    if isinstance(root_dirs, (list, tuple)):
        datasets = [
            RetinaVesselDataset(
                root_dir=rd,
                split=split,
                image_size=image_size,
                augment=augment,
                prefer_processed=prefer_processed,
                return_dict=return_dict,
            )
            for rd in root_dirs
        ]
        return ConcatDataset(datasets)
    return RetinaVesselDataset(
        root_dir=root_dirs,
        split=split,
        image_size=image_size,
        augment=augment,
        prefer_processed=prefer_processed,
        return_dict=return_dict,
    )
