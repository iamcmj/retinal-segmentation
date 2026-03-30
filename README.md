# TransUNet Retina Vessel Segmentation

## Overview
- Transformer-enhanced U-Net pipeline for retinal vessel segmentation across DRIVE, CHASE_DB1, HRF, STARE, Fundus-AVSeg, and RETA.
- Supports single-dataset runs and unified multi-dataset training with on-the-fly augmentations and optional checkpoint resume.
- Retina-only scope; no abdominal CT or multi-organ tasks.

## Environment
- Python >= 3.8 and PyTorch >= 1.10 with a matching `torchvision` build (CUDA recommended for training).
- Install dependencies from the TransUNet module:
  ```bash
  cd TransUNet
  pip install -r requirements.txt
  # or manually: torch torchvision numpy tqdm tensorboard tensorboardX ml-collections medpy SimpleITK scipy h5py
  ```
- Key scripts: `train_retina.py` (training), `test_retina.py` (inference), dataset utilities in `datasets/` and `retina_utils/`.

## Retina Dataset Layout
Place datasets under `dataset/` (relative to repo root) using the shared structure below:
```
dataset/
  DRIVE/
    training/images/*.tif|png|jpg...
    training/masks/*.tif|png|gif...
    test/images/
    test/masks/
    processed/            # optional preprocessed splits mirror training/test
  CHASE_DB1/
    training/images/
    training/masks/
    test/images/
    test/masks/
    processed/ (optional)
  HRF/
  STARE/
  Fundus-AVSeg/
  RETA/
```
- Each root (e.g., `dataset/DRIVE`) contains `training/` and `test/` subfolders with `images/` and `masks/`.
- If `processed/` exists, it is preferred automatically before raw folders.

## Expected Dataset Format
- Images: RGB fundus files (extensions among `.tif`, `.png`, `.jpg`, `.jpeg`, `.gif`).
- Masks: Single-channel vessel labels matching image stems; suffixes like `_mask`, `_manual1`, `_manual` are auto-stripped.
- Splits: `train_retina.py` uses `--train_split training` by default; `--train_split test` can target test/val-style folders if needed.

## Training
Run commands from `TransUNet/` (paths assume repo root is one level up).

### Single-Dataset Training
```bash
cd TransUNet
python train_retina.py \
  --root_path ../dataset/DRIVE \
  --train_split training \
  --img_size 512 \
  --max_epochs 150 \
  --batch_size 8
```
- Change `--root_path` to any retina dataset root (e.g., `../dataset/CHASE_DB1`).

### Unified Multi-Dataset Training (`--unified_roots`)
```bash
cd TransUNet
python train_retina.py \
  --unified_roots \
    ../dataset/DRIVE,\
    ../dataset/CHASE_DB1,\
    ../dataset/HRF,\
    ../dataset/STARE,\
    ../dataset/Fundus-AVSeg,\
    ../dataset/RETA \
  --train_split training \
  --img_size 512 \
  --max_epochs 150 \
  --batch_size 8
```
- All listed roots are merged into one training set; sampling/augmentations are applied on the fly.

## Resuming Training
Add `--resume` to load a saved checkpoint (weights and optimizer state):
```bash
python train_retina.py \
  --root_path ../dataset/DRIVE \
  --resume ../model/TU_Retina512/TU_pretrain_R50-ViT-B_16_skip3_bs24_512/epoch_100.pth
```
- When resuming unified training, keep `--unified_roots` identical to the original run.

## Inference
Use `test_retina.py` for single images or directories:
```bash
cd TransUNet
python test_retina.py \
  --image_dir ../dataset/DRIVE/test/images \
  --checkpoint ../model/TU_Retina512/TU_pretrain_R50-ViT-B_16_skip3_bs24_512/epoch_149.pth \
  --output_dir ../predictions_retina/drive_test \
  --img_size 512
```
- `--image_path` can target a single image; `--image_dir` processes a folder.
- Set `--no_resize_back` to keep masks at network resolution instead of resizing to original image size.

## Checkpoints and Logs
- Saved under `model/` (relative to repo root) with a snapshot name derived from run settings, e.g., `model/TU_Retina512/TU_pretrain_R50-ViT-B_16_skip3_bs24_512[_<dataset>]`.
- Training saves `epoch_XX.pth` in the snapshot directory (plus `log.txt` and TensorBoard logs in `log/`).
- Specify `--output_dir` during inference to store predicted masks (default `../predictions_retina`).

## Data Augmentation
- Default augmentations: random horizontal/vertical flips, small rotations, and brightness/contrast jitter applied during training.
- Disable all train-time augmentations with `--no_augment` for deterministic or ablation runs.

