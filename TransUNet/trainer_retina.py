import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_retina import RetinaVesselDataset
from utils import DiceLoss


def trainer_retina(args, model: nn.Module, snapshot_path: str) -> str:
    """
    Retina-specific trainer using 2D RGB images and masks.
    Keeps the overall training loop, optimizer, and scheduler style from trainer.py,
    but removes all 3D/Synapse-specific logic.
    """
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

    def worker_init_fn(worker_id: int) -> None:
        random.seed(args.seed + worker_id)

    train_dataset = RetinaVesselDataset(
        root_dir=args.root_path,
        split=getattr(args, "train_split", "training"),
        image_size=args.img_size,
        augment=not getattr(args, "no_augment", False),
        return_dict=False,
    )
    logging.info("The length of train set is: {}".format(len(train_dataset)))

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    bce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + "/log")
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, (image_batch, label_batch) in enumerate(trainloader):
            image_batch = image_batch.cuda() if torch.cuda.is_available() else image_batch
            label_batch = label_batch.cuda() if torch.cuda.is_available() else label_batch

            outputs = model(image_batch)
            # BCE on one-hot targets to balance classes; Dice with softmax for overlap.
            target_onehot = torch.nn.functional.one_hot(
                label_batch.long(), num_classes=num_classes
            ).permute(0, 3, 1, 2).float()
            loss_bce = bce_loss(outputs, target_onehot)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/loss_bce", loss_bce, iter_num)
            writer.add_scalar("info/loss_dice", loss_dice, iter_num)

            logging.info(
                "epoch=%d iter=%d loss=%f loss_bce=%f loss_dice=%f"
                % (epoch_num, iter_num, loss.item(), loss_bce.item(), loss_dice.item())
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
