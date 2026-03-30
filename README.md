# TransUNet Retina Vessel Segmentation

This repository adapts TransUNet for 2D retinal vessel segmentation across DRIVE, CHASE_DB1, HRF, STARE, Fundus-AVSeg, and RETA.

## What This Repo Contains
- `TransUNet/`: model definitions, training script, inference script, retina dataset loader, and model utilities
- `dataset/`: raw retinal datasets and optional preprocessed artifacts
- `model/`: downloaded ViT pretrained weights and saved training checkpoints
- `utils/`: preprocessing and visualization utilities

## Environment
This project uses `uv` for dependency and environment management.

From the repository root:

```bash
uv sync --extra tools --extra legacy --group dev
```

Recommended Python version:

```bash
cat .python-version
```

## Pretrained ViT Weights
Training expects a pretrained ViT `.npz` file under `model/vit_checkpoint/imagenet21k/`.

For the default `R50-ViT-B_16` setup:

```bash
mkdir -p model/vit_checkpoint/imagenet21k
curl -L "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz" \
  -o "model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
```

Other supported model names in `TransUNet/networks/vit_seg_configs.py` may require:
- `ViT-B_16.npz`
- `ViT-B_32.npz`
- `ViT-L_16.npz`

## Dataset Layout
Place datasets under `dataset/` using the following structure:

```text
dataset/
  DRIVE/
    training/images/
    training/masks/
    test/images/
    test/masks/
  CHASE_DB1/
    training/images/
    training/masks/
    test/images/
    test/masks/
  HRF/
  STARE/
  Fundus-AVSeg/
  RETA/
```

Expected format:
- images: `.tif`, `.png`, `.jpg`, `.jpeg`, `.ppm`, `.gif`
- masks: matching stems, with suffixes like `_mask`, `_manual1`, `_manual`, `_1st_manual` handled automatically by the retina dataset loader

## Preprocessing
Optional preprocessing lives in `utils/preprocess.py`.

It:
- reads raw `training/` and `test/` images/masks
- applies green-channel CLAHE and mask binarization
- writes `.npy` artifacts into `dataset/<name>/processed/<split>/`

Run it from the repository root:

```bash
uv run python utils/preprocess.py
```

Note:
- preprocessing is optional
- the current training loader still reads image files directly, not `.npy` caches

## Training
Run training commands from `TransUNet/` so relative paths resolve correctly.

### Single-Dataset Training
```bash
cd TransUNet
uv run python train_retina.py \
  --root_path ../dataset/DRIVE \
  --train_split training \
  --img_size 512 \
  --max_epochs 150 \
  --batch_size 8
```

### Unified Multi-Dataset Training
```bash
cd TransUNet
uv run python train_retina.py \
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

## Resume
Resume currently reloads model weights from a saved checkpoint path.

```bash
cd TransUNet
uv run python train_retina.py \
  --root_path ../dataset/DRIVE \
  --resume ../model/TU_Retina512/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_512_unified/epoch_99.pth
```

Note:
- this is currently weight loading, not full optimizer/epoch state restoration

## Inference
Inference is handled by `TransUNet/test_retina.py`.

### Directory Inference
```bash
cd TransUNet
uv run python test_retina.py \
  --image_dir ../dataset/DRIVE/test/images \
  --checkpoint ../model/TU_Retina512/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_512_unified/epoch_149.pth \
  --output_dir ../predictions_retina/drive_test \
  --img_size 512
```

### Single Image Inference
```bash
cd TransUNet
uv run python test_retina.py \
  --image_path ../dataset/DRIVE/test/images/01_test.tif \
  --checkpoint ../model/TU_Retina512/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_512_unified/epoch_149.pth
```

Inference outputs `*_mask.png` files to the target output directory.

## Checkpoints
Training outputs are saved under `model/`, for example:

```text
model/TU_Retina512/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_512_unified/
  epoch_99.pth
  epoch_149.pth
  log.txt
  log/
```

## Project Status Notes
- `TransUNet/train_retina.py` is the active training entrypoint
- `TransUNet/test_retina.py` is the active inference entrypoint
- `TransUNet/trainer_retina.py` exists but is not the active training path
- `TransUNet/legacy_synapse_viewer.py` is a legacy Synapse example and not part of the retina workflow

## References
- TransUNet paper: https://arxiv.org/pdf/2102.04306.pdf
- Google ViT: https://github.com/google-research/vision_transformer
