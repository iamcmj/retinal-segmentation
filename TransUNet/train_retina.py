import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_retina import RetinaVesselDataset as RetinaDataset, build_retina_dataset
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from utils import DiceLoss

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../data/Retina", help="root dir for retina data")
parser.add_argument("--dataset", type=str, default="Retina", help="experiment_name")
parser.add_argument("--num_classes", type=int, default=2, help="output channel of network")
parser.add_argument("--train_split", type=str, default="training", help="sub-folder under root_path containing images/masks")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum iteration number to train")
parser.add_argument("--max_epochs", type=int, default=150, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--img_size", type=int, default=512, help="input patch size of network input")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--n_skip", type=int, default=3, help="using number of skip-connect, default is num")
parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16", help="select one vit model")
parser.add_argument("--vit_patches_size", type=int, default=16, help="vit_patches_size, default is 16")
parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
parser.add_argument("--no_augment", action="store_true", help="disable train-time augmentations")
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="path to a checkpoint .pth to resume model weights from before training",
)
parser.add_argument(
    "--root_paths",
    type=str,
    default=None,
    help="comma-separated list of dataset roots to train sequentially (overrides --root_path)",
)
parser.add_argument(
    "--unified_roots",
    type=str,
    default=None,
    help="comma-separated list of dataset roots to merge into a single training dataset",
)
args = parser.parse_args()


def trainer_retina(args, model, snapshot_path, device):
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataset_roots = args.unified_roots if args.unified_roots else args.root_path
    train_dataset = build_retina_dataset(
        root_dirs=dataset_roots,
        split=args.train_split,
        image_size=args.img_size,
        augment=not args.no_augment,
    )
    logging.info("The length of train set is: {}".format(len(train_dataset)))

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + "/log")
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    dataset_label = getattr(args, "current_dataset_label", "")
    iterator = tqdm(
        range(max_epoch),
        ncols=70,
        desc=f"Dataset={dataset_label} epochs" if dataset_label else "epochs",
    )

    for epoch_num in iterator:
        for i_batch, (image_batch, label_batch) in enumerate(trainloader):
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/loss_ce", loss_ce, iter_num)
            logging.info(
                "dataset=%s epoch=%d iter=%d loss=%f loss_ce=%f"
                % (
                    dataset_label if dataset_label else "N/A",
                    epoch_num,
                    iter_num,
                    loss.item(),
                    loss_ce.item(),
                )
            )

            if iter_num % 20 == 0:
                img = image_batch[0]
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                writer.add_image("train/Image", img, iter_num)

                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True).float()
                writer.add_image("train/Prediction", preds[0] * 255.0, iter_num)

                labs = label_batch.unsqueeze(1).float()
                writer.add_image("train/GroundTruth", labs[0] * 255.0, iter_num)

        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def _build_snapshot_path(args, dataset_suffix: str | None = None) -> str:
    snapshot_path = "../model/{}/{}".format(args.exp, "TU")
    snapshot_path = snapshot_path + "_pretrain" if args.is_pretrain else snapshot_path
    snapshot_path += "_" + args.vit_name
    snapshot_path = snapshot_path + "_skip" + str(args.n_skip)
    snapshot_path = (
        snapshot_path + "_vitpatch" + str(args.vit_patches_size)
        if args.vit_patches_size != 16
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_" + str(args.max_iterations)[0:2] + "k" if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + "_epo" + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + "_bs" + str(args.batch_size)
    snapshot_path = snapshot_path + "_lr" + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + "_" + str(args.img_size)
    snapshot_path = snapshot_path + "_s" + str(args.seed) if args.seed != 1234 else snapshot_path
    if dataset_suffix:
        snapshot_path = snapshot_path + "_" + dataset_suffix
    return snapshot_path


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
            "root_path": "../data/Retina",
            "num_classes": 2,
            "train_split": "training",
        },
    }
    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    args.num_classes = dataset_config[dataset_name]["num_classes"]
    # If multiple roots are provided, honor them sequentially; otherwise use single root_path.
    root_paths = (
        [p.strip() for p in args.root_paths.split(",") if p.strip()]
        if args.root_paths
        else [args.root_path or dataset_config[dataset_name]["root_path"]]
    )
    args.train_split = args.train_split or dataset_config[dataset_name]["train_split"]
    args.is_pretrain = True
    args.exp = "TU_" + dataset_name + str(args.img_size)

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
    net.load_from(weights=np.load(config_vit.pretrained_path))

    # Optional: resume from a previous checkpoint.
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device)
        net.load_state_dict(state_dict)
        print(f"Loaded checkpoint weights from {args.resume}")

    # Unified training across multiple roots as one dataset.
    if args.unified_roots:
        unified_list = [p.strip() for p in args.unified_roots.split(",") if p.strip()]
        args.unified_roots = unified_list
        args.current_dataset_label = "unified"
        dataset_suffix = "unified"
        snapshot_path = _build_snapshot_path(args, dataset_suffix=dataset_suffix)
        os.makedirs(snapshot_path, exist_ok=True)
        print(f"[Unified] Training on merged roots={unified_list} -> snapshots: {snapshot_path}")
        trainer_retina(args, net, snapshot_path, device)
    else:
        # Sequential training across multiple roots (e.g., DRIVE, CHASE_DB1, HRF, STARE).
        for idx, root_path in enumerate(root_paths, 1):
            args.root_path = root_path
            args.current_dataset_label = Path(root_path).name
            dataset_suffix = Path(root_path).name if len(root_paths) > 1 else None
            snapshot_path = _build_snapshot_path(args, dataset_suffix=dataset_suffix)
            os.makedirs(snapshot_path, exist_ok=True)
            print(
                f"[{idx}/{len(root_paths)}] Training on dataset={args.current_dataset_label} "
                f"root_path={root_path} -> snapshots: {snapshot_path}"
            )
            trainer_retina(args, net, snapshot_path, device)
