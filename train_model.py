import copy
import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms

from src.pcb_pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_INPUT_SIZE,
    build_model,
)

#  CONFIG
DATASET_ROOT          = "dataset"
MODEL_PATH            = "pcb_defect_model.pth"
BATCH_SIZE            = 16
EPOCHS                = 20    # ↑ was 25 — more epochs = better learning
FREEZE_BACKBONE_EPOCHS= 4       # ↑ was 5  — longer freeze = better features
LEARNING_RATE         = 3e-4
WEIGHT_DECAY          = 1e-4
VAL_SPLIT             = 0.2
SEED                  = 42
NUM_WORKERS           = 0
LABEL_SMOOTHING       = 0.10     # ↑ was 0.05 — reduces overconfidence on dominant class


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#  TRANSFORMS
#  Heavier augmentation helps minority classes
#  generalise from limited samples
def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(MODEL_INPUT_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.25,  
            contrast=0.25,  
            saturation=0.15,  
            hue=0.05,          
        ),
        transforms.RandomGrayscale(p=0.05),                                
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),              
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform
def make_balanced_sampler(dataset):
    """Return a WeightedRandomSampler that gives equal probability to each class."""
    # get label for every sample
    if hasattr(dataset, "dataset"):
        # it's a Subset
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        labels = dataset.targets

    counts      = Counter(labels)
    n_classes   = len(counts)
    total       = len(labels)

    # weight per sample = 1 / count_of_its_class
    class_weight = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}
    sample_weights = [class_weight[lbl] for lbl in labels]

    print("\nClass weights for sampler:")
    for cls, w in class_weight.items():
        print(f"  class {cls}: count={counts[cls]}, weight={w:.4f}")

    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )
def resolve_datasets(dataset_root: str):
    root        = Path(dataset_root)
    train_dir   = root / "train"
    val_dir     = root / "val"
    test_dir    = root / "test"

    train_transform, eval_transform = build_transforms()

    if train_dir.exists():
        full_train  = datasets.ImageFolder(train_dir, transform=train_transform)
        class_names = full_train.classes

        if val_dir.exists():
            val_dataset   = datasets.ImageFolder(val_dir, transform=eval_transform)
            train_dataset = full_train
        else:
            eval_copy  = datasets.ImageFolder(train_dir, transform=eval_transform)
            val_size   = max(1, int(len(full_train) * VAL_SPLIT))
            train_size = len(full_train) - val_size
            generator  = torch.Generator().manual_seed(SEED)

            train_indices, val_indices = random_split(
                range(len(full_train)), [train_size, val_size], generator=generator
            )
            train_dataset = torch.utils.data.Subset(full_train, train_indices.indices)
            val_dataset   = torch.utils.data.Subset(eval_copy,  val_indices.indices)

        test_dataset = None
        if test_dir.exists():
            test_candidate = datasets.ImageFolder(test_dir, transform=eval_transform)
            if test_candidate.classes == class_names:
                test_dataset = test_candidate

        return train_dataset, val_dataset, test_dataset, class_names

    # fallback — flat root
    full_dataset = datasets.ImageFolder(root, transform=train_transform)
    class_names  = full_dataset.classes
    eval_copy    = datasets.ImageFolder(root, transform=eval_transform)

    val_size   = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    generator  = torch.Generator().manual_seed(SEED)

    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
    val_dataset   = torch.utils.data.Subset(eval_copy,    val_indices.indices)

    return train_dataset, val_dataset, None, class_names
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    # per-class tracking
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            preds    = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # per-class accuracy
    per_class = {}
    for lbl, pred in zip(all_labels, all_preds):
        if lbl not in per_class:
            per_class[lbl] = {"correct": 0, "total": 0}
        per_class[lbl]["total"]   += 1
        per_class[lbl]["correct"] += int(pred == lbl)

    return total_loss / max(total, 1), correct / max(total, 1), per_class
def main():
    set_seed(SEED)

    train_dataset, val_dataset, test_dataset, class_names = resolve_datasets(DATASET_ROOT)

    print(f"\nClasses found: {class_names}")
    print(f"Samples → Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Class distribution ──────────────────────────────────────────────────
    if hasattr(train_dataset, "dataset"):
        labels = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    else:
        labels = train_dataset.targets

    counts = Counter(labels)
    print("\nTraining class distribution:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {counts.get(i, 0)} images")

    # ── Balanced sampler — fixes class imbalance ────────────────────────────
    sampler = make_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,     # replaces shuffle=True
        num_workers = NUM_WORKERS,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = build_model(len(class_names)).to(device)

    # ── Class-weighted loss (double protection against imbalance) ───────────
    total  = sum(counts.values())
    weights = torch.tensor(
        [total / counts.get(i, 1) for i in range(len(class_names))],
        dtype=torch.float,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_state, best_val_acc = None, 0.0

    print(f"\n{'─'*60}")
    print(f"Starting training for {EPOCHS} epochs")
    print(f"{'─'*60}\n")

    for epoch in range(EPOCHS):
        if epoch == FREEZE_BACKBONE_EPOCHS:
            print(f"\n🔓 Unfreezing backbone at epoch {epoch+1}...")
            for param in model.features.parameters():
                param.requires_grad = True

            # lower LR for fine-tuning backbone
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr           = LEARNING_RATE * 0.1,
                weight_decay = WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - FREEZE_BACKBONE_EPOCHS
            )
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        train_acc  = correct / total
        train_loss = total_loss / total

        val_loss, val_acc, per_class = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}   | val_acc={val_acc:.4f}"
        )

        # print per-class accuracy every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("  Per-class val accuracy:")
            for i, name in enumerate(class_names):
                stats = per_class.get(i, {"correct": 0, "total": 0})
                acc   = stats["correct"] / max(stats["total"], 1)
                print(f"    {name:20s}: {acc:.4f}  ({stats['total']} samples)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            print(f"New best val_acc: {best_val_acc:.4f} — checkpoint saved")

    # ── Save best model ─────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save({
        "state_dict" : model.state_dict(),
        "classes"    : class_names,
        "mean"       : IMAGENET_MEAN,
        "std"        : IMAGENET_STD,
    }, MODEL_PATH)

    print(f"\n{'─'*60}")
    print(f"Model saved to {MODEL_PATH}")
    print(f"   Best val_acc = {best_val_acc:.4f}")
    print(f"{'─'*60}\n")
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        test_loss, test_acc, per_class = evaluate(model, test_loader, criterion, device)
        print(f"Test accuracy: {test_acc:.4f}")
        print("Per-class test accuracy:")
        for i, name in enumerate(class_names):
            stats = per_class.get(i, {"correct": 0, "total": 0})
            acc   = stats["correct"] / max(stats["total"], 1)
            print(f"  {name:20s}: {acc:.4f}  ({stats['total']} samples)")


if __name__ == "__main__":
    main()
