from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

MASK_EXTS = {".png", ".tif", ".tiff", ".gif", ".jpg", ".jpeg"}
MASK_SUFFIXES = ["_mask", "_manual1", "_manual", "_1st_manual", "_manual_1"]

def build_mask_index(mask_dir: Path):
    mask_index = {}
    for mask_path in sorted(mask_dir.iterdir()):
        if not mask_path.is_file() or mask_path.suffix.lower() not in MASK_EXTS:
            continue

        stem = mask_path.stem.lower()
        candidate_keys = {stem}
        for suffix in MASK_SUFFIXES:
            if stem.endswith(suffix):
                candidate_keys.add(stem[: -len(suffix)])

        for key in candidate_keys:
            mask_index.setdefault(key, mask_path)
    return mask_index

def find_mask(mask_index, stem: str):
    return mask_index.get(stem.lower())

def load_image(path: Path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def green_channel_clahe(img):
    green = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)
    enhanced = np.stack([enhanced]*3, axis=2)
    return enhanced

def preprocess_image(path, size):
    img = load_image(path)
    img = green_channel_clahe(img)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

def preprocess_mask(path, size):
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, Image.NEAREST)
    mask = np.array(mask, dtype=np.uint8)
    return (mask > 20).astype(np.float32)

def process_dataset(root: Path, size):
    tqdm.write(f"\n========== Processing {root} ==========")

    for subset in ["training", "test"]:
        subset_dir = root / subset
        if not subset_dir.exists():
            tqdm.write(f"Skipping {subset_dir} (does not exist)")
            continue

        img_dir = subset_dir / "images"
        mask_dir = subset_dir / "1st_manual"
        if not mask_dir.exists():
            mask_dir = subset_dir / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            tqdm.write(f"Missing images or masks in {subset_dir}, skipping.")
            continue

        out_base = root / "processed" / subset
        out_img = out_base / "images"
        out_mask = out_base / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)

        images = sorted(path for path in img_dir.glob("*") if path.is_file())
        mask_index = build_mask_index(mask_dir)
        count = 0
        skipped = 0

        for img_path in tqdm(
            images,
            desc=f"{root.name}/{subset}",
            unit="img",
            dynamic_ncols=True,
            leave=False,
        ):
            stem = img_path.stem
            mask_path = find_mask(mask_index, stem)
            if mask_path is None:
                tqdm.write(f"  mask not found for {img_path.name}")
                continue

            out_img_path = out_img / f"{stem}.npy"
            out_mask_path = out_mask / f"{stem}.npy"
            if out_img_path.exists() and out_mask_path.exists():
                skipped += 1
                continue

            img = preprocess_image(img_path, size)
            mask = preprocess_mask(mask_path, size)

            np.save(out_img_path, img)
            np.save(out_mask_path, mask)
            count += 1

        tqdm.write(f"{subset}: processed {count} image-mask pairs, skipped {skipped}.")

def main():
    dataset_root = Path("dataset")
    size = (256, 256)

    dataset_dirs = sorted(folder for folder in dataset_root.iterdir() if folder.is_dir())
    for folder in tqdm(dataset_dirs, desc="Datasets", unit="dataset", dynamic_ncols=True):
        process_dataset(folder, size)

if __name__ == "__main__":
    main()
