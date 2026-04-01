"""Training entrypoint for retinal vessel segmentation with TransUNet."""

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
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from datasets.dataset_retina import RetinaVesselDataset as RetinaDataset, build_retina_dataset
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from utils_retina import DiceLoss

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
    "--val_split",
    type=str,
    default=None,
    help="validation split name; if omitted, tries validation/val/test in that order",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="path to a checkpoint .pth to resume full training state or load legacy model weights",
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


def _build_logger(snapshot_path: str):
    """Create an isolated logger for a single training run and snapshot directory."""
    logger_name = f"train_retina.{Path(snapshot_path).resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Rebuild handlers on every call so sequential training runs do not reuse
    # file/stdout handlers from a previous dataset.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(snapshot_path, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    """Close and detach all handlers owned by a training logger."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped with DataParallel."""
    return model.module if hasattr(model, "module") else model


def _normalize_model_state_dict(state_dict):
    """Strip a leading DataParallel 'module.' prefix when present."""
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _move_optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    """Move optimizer tensor state to the active device after loading a checkpoint."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _save_training_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch_num: int,
    iter_num: int,
    args,
    save_path: str,
    best_val_loss: float | None = None,
) -> None:
    """Save a full training checkpoint that can be used for true resume."""
    checkpoint = {
        "model_state": _unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch_num,
        "iter_num": iter_num,
        "args": vars(args).copy(),
    }
    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss
    torch.save(checkpoint, save_path)


def _load_training_checkpoint(
    resume_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
):
    """Load either a full training checkpoint or a legacy weights-only checkpoint."""
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    target_model = _unwrap_model(model)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        target_model.load_state_dict(_normalize_model_state_dict(checkpoint["model_state"]))

        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            _move_optimizer_state_to_device(optimizer, device)

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        iter_num = int(checkpoint.get("iter_num", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        logger.info(
            "Loaded full checkpoint from %s and will resume at epoch=%d, iter=%d, best_val_loss=%s",
            resume_path,
            start_epoch,
            iter_num,
            "inf" if best_val_loss == float("inf") else f"{best_val_loss:.6f}",
        )
        return start_epoch, iter_num, best_val_loss

    legacy_state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    target_model.load_state_dict(_normalize_model_state_dict(legacy_state_dict))
    logger.warning(
        "Loaded legacy weights-only checkpoint from %s. "
        "Optimizer, epoch, and iteration state were not restored.",
        resume_path,
    )
    return 0, 0, float("inf")


def _validation_split_candidates(preferred_split: str | None) -> list[str]:
    """Return the ordered list of split names to try for validation data."""
    candidates: list[str] = []
    if preferred_split:
        candidates.append(preferred_split)
    for split_name in ("validation", "val", "test"):
        if split_name not in candidates:
            candidates.append(split_name)
    return candidates


def _build_single_validation_dataset(
    root_dir: str | Path,
    image_size: int,
    candidate_splits: list[str],
):
    """Build the first available validation dataset for one root directory."""
    for split_name in candidate_splits:
        try:
            dataset = RetinaDataset(
                root_dir=root_dir,
                split=split_name,
                image_size=image_size,
                augment=False,
                return_dict=False,
            )
            return dataset, split_name
        except (FileNotFoundError, RuntimeError):
            continue
    return None, None


def _build_validation_dataset(root_dirs, image_size: int, preferred_split: str | None, logger: logging.Logger):
    """Build a validation dataset from one root or multiple dataset roots."""
    candidate_splits = _validation_split_candidates(preferred_split)
    if isinstance(root_dirs, (list, tuple)):
        datasets = []
        selected_splits = []
        for root_dir in root_dirs:
            dataset, split_name = _build_single_validation_dataset(root_dir, image_size, candidate_splits)
            if dataset is not None:
                datasets.append(dataset)
                selected_splits.append(f"{Path(root_dir).name}:{split_name}")
        if not datasets:
            logger.warning("No validation split was found for any dataset root. Validation will be skipped.")
            return None, []
        return (datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)), selected_splits

    dataset, split_name = _build_single_validation_dataset(root_dirs, image_size, candidate_splits)
    if dataset is None:
        logger.warning("No validation split was found under %s. Validation will be skipped.", root_dirs)
        return None, []
    return dataset, [f"{Path(root_dirs).name}:{split_name}"]


def _validate_retina(
    model: nn.Module,
    valloader: DataLoader,
    device: torch.device,
    ce_loss: nn.Module,
    dice_loss: nn.Module,
):
    """Compute average validation loss using the same objective as training."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for image_batch, label_batch in valloader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_dice += loss_dice.item()
            num_batches += 1

    if was_training:
        model.train()

    if num_batches == 0:
        return None

    return {
        "loss": total_loss / num_batches,
        "loss_ce": total_ce / num_batches,
        "loss_dice": total_dice / num_batches,
    }


def trainer_retina(args, model, snapshot_path, device):
    """Run retina training for one dataset root or one unified merged dataset."""
    logger = _build_logger(snapshot_path)
    writer = None
    try:
        logger.info(str(args))

        base_lr = args.base_lr
        num_classes = args.num_classes
        batch_size = args.batch_size * args.n_gpu

        def worker_init_fn(worker_id):
            random.seed(args.seed + worker_id)

        # Unified training passes a list of roots; single-dataset training uses
        # one root_path. The dataset builder handles both cases.
        dataset_roots = args.unified_roots if args.unified_roots else args.root_path
        train_dataset = build_retina_dataset(
            root_dirs=dataset_roots,
            split=args.train_split,
            image_size=args.img_size,
            augment=not args.no_augment,
        )
        logger.info("The length of train set is: {}".format(len(train_dataset)))

        trainloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        val_dataset, val_sources = _build_validation_dataset(
            dataset_roots,
            image_size=args.img_size,
            preferred_split=args.val_split,
            logger=logger,
        )
        valloader = None
        if val_dataset is not None:
            valloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            logger.info("Validation dataset sources: %s", ", ".join(val_sources))
            logger.info("The length of val set is: %d", len(val_dataset))

        if args.n_gpu > 1:
            model = nn.DataParallel(model)
        model.train()

        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(num_classes)
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        start_epoch = 0
        iter_num = 0
        best_val_loss = float("inf")
        if args.resume:
            # Full checkpoints restore model, optimizer, epoch, and iteration state.
            # Older checkpoints still load as weights-only for backward compatibility.
            start_epoch, iter_num, best_val_loss = _load_training_checkpoint(
                args.resume,
                model,
                optimizer,
                device,
                logger,
            )

        writer = SummaryWriter(snapshot_path + "/log")
        max_epoch = args.max_epochs
        max_iterations = args.max_epochs * len(trainloader)
        logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
        dataset_label = getattr(args, "current_dataset_label", "")
        if start_epoch >= max_epoch:
            logger.info(
                "Resume checkpoint is already at or beyond max_epochs (%d >= %d). Nothing to train.",
                start_epoch,
                max_epoch,
            )
            return "Training Finished!"

        iterator = tqdm(
            range(start_epoch, max_epoch),
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

                # Keep the original polynomial decay schedule, but continue it from
                # the restored iteration counter when resuming.
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_

                iter_num += 1
                writer.add_scalar("info/lr", lr_, iter_num)
                writer.add_scalar("info/total_loss", loss, iter_num)
                writer.add_scalar("info/loss_ce", loss_ce, iter_num)
                logger.info(
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
            if valloader is not None:
                val_metrics = _validate_retina(model, valloader, device, ce_loss, dice_loss)
                if val_metrics is not None:
                    writer.add_scalar("val/loss", val_metrics["loss"], epoch_num + 1)
                    writer.add_scalar("val/loss_ce", val_metrics["loss_ce"], epoch_num + 1)
                    writer.add_scalar("val/loss_dice", val_metrics["loss_dice"], epoch_num + 1)
                    logger.info(
                        "dataset=%s epoch=%d val_loss=%f val_loss_ce=%f val_loss_dice=%f",
                        dataset_label if dataset_label else "N/A",
                        epoch_num,
                        val_metrics["loss"],
                        val_metrics["loss_ce"],
                        val_metrics["loss_dice"],
                    )
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        best_model_path = os.path.join(snapshot_path, "best_model.pth")
                        _save_training_checkpoint(
                            model,
                            optimizer,
                            epoch_num,
                            iter_num,
                            args,
                            best_model_path,
                            best_val_loss=best_val_loss,
                        )
                        logger.info(
                            "save best model to %s (best_val_loss=%f)",
                            best_model_path,
                            best_val_loss,
                        )

            if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
                save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
                _save_training_checkpoint(
                    model,
                    optimizer,
                    epoch_num,
                    iter_num,
                    args,
                    save_mode_path,
                    best_val_loss=best_val_loss,
                )
                logger.info("save model to {}".format(save_mode_path))

            if epoch_num >= max_epoch - 1:
                save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
                _save_training_checkpoint(
                    model,
                    optimizer,
                    epoch_num,
                    iter_num,
                    args,
                    save_mode_path,
                    best_val_loss=best_val_loss,
                )
                logger.info("save model to {}".format(save_mode_path))
                iterator.close()
                break
        return "Training Finished!"
    finally:
        if writer is not None:
            writer.close()
        _close_logger(logger)


def _build_snapshot_path(args, dataset_suffix: str | None = None) -> str:
    """Build the checkpoint directory name used by training and inference scripts."""
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
    # Match the original script behavior: deterministic mode disables the faster
    # cuDNN autotuner for reproducibility.
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

    # Unified training merges all roots into a single concatenated dataset.
    if args.unified_roots:
        unified_list = [p.strip() for p in args.unified_roots.split(",") if p.strip()]
        args.unified_roots = unified_list
        args.current_dataset_label = "unified"
        dataset_suffix = "unified"
        snapshot_path = _build_snapshot_path(args, dataset_suffix=dataset_suffix)
        os.makedirs(snapshot_path, exist_ok=True)
        print(f"[Unified] Training on merged roots={unified_list} -> snapshots: {snapshot_path}")
        trainer_retina(args, net, snapshot_path, device)
        args.resume = None
    else:
        # Sequential training reuses the same model instance and continues training
        # dataset by dataset across multiple roots.
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
            args.resume = None
