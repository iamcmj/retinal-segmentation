import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from datasets.dataset_retina import VALID_EXTENSIONS
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from retina_utils.test_single_image import test_single_image

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default=None, help="single image path or directory of images")
parser.add_argument("--image_dir", type=str, default=None, help="directory containing test images (overrides image_path if set)")
parser.add_argument("--output_dir", type=str, default="../predictions_retina", help="where to save predicted PNG masks")
parser.add_argument("--checkpoint", type=str, default=None, help="path to a trained checkpoint; defaults to snapshot naming")

parser.add_argument("--dataset", type=str, default="Retina", help="experiment_name")
parser.add_argument("--num_classes", type=int, default=2, help="output channel of network")
parser.add_argument("--img_size", type=int, default=512, help="input patch size of network input")
parser.add_argument("--max_iterations", type=int, default=30000, help="kept for snapshot name compatibility")
parser.add_argument("--max_epochs", type=int, default=150, help="kept for snapshot name compatibility")
parser.add_argument("--batch_size", type=int, default=24, help="kept for snapshot name compatibility")
parser.add_argument("--n_skip", type=int, default=3, help="using number of skip-connect, default is num")
parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16", help="select one vit model")
parser.add_argument("--vit_patches_size", type=int, default=16, help="vit_patches_size, default is 16")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic inference")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--base_lr", type=float, default=0.01, help="kept for snapshot name compatibility")
parser.add_argument("--no_resize_back", action="store_true", help="keep mask at network resolution instead of original size")
args = parser.parse_args()


def _gather_images(paths: Iterable[Path]) -> List[Path]:
    valid_paths: List[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            valid_paths.append(path)
    return valid_paths


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "Retina": {
            "num_classes": 2,
        },
    }
    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.is_pretrain = True
    args.exp = "TU_" + dataset_name + str(args.img_size)

    snapshot_path = "../model/{}/{}".format(args.exp, "TU")
    snapshot_path = snapshot_path + "_pretrain" if args.is_pretrain else snapshot_path
    snapshot_path += "_" + args.vit_name
    snapshot_path = snapshot_path + "_skip" + str(args.n_skip)
    snapshot_path = (
        snapshot_path + "_vitpatch" + str(args.vit_patches_size)
        if args.vit_patches_size != 16
        else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_" + str(args.max_iterations)[0:2] + "k"
        if args.max_iterations != 30000
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_epo" + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + "_bs" + str(args.batch_size)
    snapshot_path = snapshot_path + "_lr" + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + "_" + str(args.img_size)
    snapshot_path = snapshot_path + "_s" + str(args.seed) if args.seed != 1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)

    if args.checkpoint is None:
        snapshot = os.path.join(snapshot_path, "best_model.pth")
        if not os.path.exists(snapshot):
            snapshot = snapshot.replace("best_model", "epoch_" + str(args.max_epochs - 1))
    else:
        snapshot = args.checkpoint
    net.load_state_dict(torch.load(snapshot, map_location=device))
    snapshot_name = Path(snapshot_path).name

    log_folder = "./test_log_retina/test_log_" + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.image_dir is not None:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
        candidate_paths = image_dir.iterdir()
    elif args.image_path is not None:
        image_path = Path(args.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path not found: {args.image_path}")
        candidate_paths = image_path.iterdir() if image_path.is_dir() else [image_path]
    else:
        raise ValueError("Provide --image_dir or --image_path to run inference.")

    image_paths = _gather_images(candidate_paths)
    if not image_paths:
        raise RuntimeError("No valid images found for inference.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net.eval()
    for image_path in tqdm(image_paths, desc="Infer"):
        save_path = output_dir / (image_path.stem + "_mask.png")
        mask_np = test_single_image(
            net,
            str(image_path),
            img_size=args.img_size,
            device=device,
            save_path=str(save_path),
            resize_back=not args.no_resize_back,
        )
        logging.info("Saved mask for %s to %s (shape=%s)", image_path.name, save_path, mask_np.shape)

    logging.info("Inference complete. Masks saved to %s", output_dir)
