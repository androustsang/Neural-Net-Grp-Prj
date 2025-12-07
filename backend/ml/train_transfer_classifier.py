"""
train_transfer_classifier.py

End-to-end transfer-learning training script for binary pothole classification.

Key features:
- Device auto-detection (CUDA / MPS / CPU)
- macOS multiprocessing-safe DataLoader config
- Transfer learning (EfficientNet-B0, ResNet34, MobileNetV3)
- Weighted sampling + pos_weight or FocalLoss for class imbalance
- On-the-fly augmentations (torchvision)
- AMP (only if CUDA available)
- LR scheduler, early stopping, checkpointing, threshold tuning

Usage:
    python3 train_transfer_classifier.py

Files required:
- utils.py in same package, exposing ClassifierDataset that returns (img_tensor [3,H,W], label)
"""

import os
import time
import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve

# Import your robust dataset from utils.py (assumes it's in same folder or in PYTHONPATH)
from utils import ClassifierDataset


# -------------------------
# CONFIG - edit as needed
# -------------------------
# Data paths (update as required)
TRAIN_TXT = "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train/_annotations.txt"
TRAIN_ROOT = "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train"
VAL_TXT = "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid/_annotations.txt"
VAL_ROOT = "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid"
TEST_TXT = "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test/_annotations.txt"
TEST_ROOT = "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test"

# Image size. Use (640,640) to preserve detail exported by Roboflow, or (224,224) to save memory.
IMG_SIZE = (640, 640)  # (H, W)

# Backbone choices: "efficientnet_b0", "resnet34", "mobilenet_v3_large"
BACKBONE = "efficientnet_b0"
PRETRAINED = True
FEATURE_EXTRACT = False  # if True: freeze most backbone parameters

# Training hyperparams
BATCH_SIZE = 8       # reduce if OOM on GPU
LR = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30

# Scheduler & early-stopping
PATIENCE = 6
LR_REDUCE_PATIENCE = 3
MIN_IMPROVEMENT = 1e-4

# Checkpoint
MODEL_SAVE_PATH = "backend/ml/models/pothole_tl_best.pth"

# Use focal loss? Set to True to enable focal loss (may help imbalance)
USE_FOCAL_LOSS = False
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 2.0

# DataLoader worker config (will be auto-adjusted for macOS / GPU)
DEFAULT_NUM_WORKERS_GPU = 4
DEFAULT_NUM_WORKERS_CPU = 0


# -------------------------
# Utilities: device, workers
# -------------------------
def get_device():
    """
    Choose device: cuda > mps (Apple) > cpu
    Output:
        device (torch.device)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def dataloader_worker_settings(device):
    """
    Returns num_workers, pin_memory, persistent_workers based on device and OS.
    Input:
        device (torch.device)
    Output:
        num_workers (int), pin_memory (bool), persistent_workers (bool)
    """
    if device.type == "cuda":
        return DEFAULT_NUM_WORKERS_GPU, True, True
    # For MPS (mac GPU) or CPU training, prefer safer settings on macOS:
    # macOS has spawn-based multiprocessing; to avoid issues set workers=0 unless user wants parallelism.
    # We'll default to 0 to be safe; user can edit constants above to increase.
    return DEFAULT_NUM_WORKERS_CPU, False, False


# -------------------------
# Focal Loss implementation
# -------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (works with logits).
    Input:
        logits (Tensor): shape (N,1)
        targets (Tensor): shape (N,1) floats 0/1
    Output:
        scalar loss
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        mod = (1 - p_t) ** self.gamma
        loss = self.alpha * mod * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -------------------------
# Build model with binary head
# -------------------------
def build_model(backbone_name="efficientnet_b0", pretrained=True, feature_extract=False):
    """
    Build model with final head producing single logit (binary classification).

    Input:
        backbone_name (str): one of "efficientnet_b0", "resnet34", "mobilenet_v3_large"
        pretrained (bool): load ImageNet weights
        feature_extract (bool): If True, freeze most backbone params

    Output:
        model (nn.Module)
    """
    if backbone_name == "efficientnet_b0":
        # EfficientNet in torchvision has different naming for weights; use built-in API
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        # replace classifier with dropout + linear -> single logit
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, 1))
    elif backbone_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    elif backbone_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[0].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
        # classifier often (Linear) or sequential; replace with single logit
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 1))
    else:
        raise ValueError("Unsupported backbone: " + backbone_name)

    if feature_extract:
        # Freeze parameters except final layer
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Unfreeze classifier / head
        if backbone_name.startswith("efficientnet"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif backbone_name.startswith("resnet"):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif backbone_name.startswith("mobilenet"):
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model


# -------------------------
# Transforms and dataset wrapping
# -------------------------
def get_transforms(img_size):
    """
    Input:
        img_size (tuple): (H, W)
    Output:
        train_transforms (torchvision.transforms.Compose),
        val_transforms (torchvision.transforms.Compose)
    Note:
        This expects input images already as torch.Tensor in [0,1], CHW (your ClassifierDataset does that).
        torchvision transforms can operate on tensors.
    """
    train_transforms = transforms.Compose([
        # Spatial transforms that accept tensor (works on torch >=1.7). If transform complains, convert to PIL in dataset.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.25),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms


class WrappedDataset(torch.utils.data.Dataset):
    """
    Wrap a base dataset (that yields (tensor_img [3,H,W] in [0,1], label))
    and apply torchvision transforms which accept tensors.

    Input:
        base_dataset: instance of ClassifierDataset
        transform: torchvision transforms.Compose or None
    Output (__getitem__):
        (img_transformed, label_tensor)
    """
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]  # img is torch.Tensor [3,H,W], 0..1
        if self.transform is not None:
            # torchvision transforms operate on tensor in C,H,W
            img = self.transform(img)
        return img, label


# -------------------------
# Helpers to compute class counts & sampler
# -------------------------
def extract_has_hole_list(dataset):
    """
    Extract list of binary labels from dataset.
    Works if dataset has .data (list of dicts) or .df (pandas).
    """
    if hasattr(dataset, "data"):
        return [int(item["has_hole"]) for item in dataset.data]
    if hasattr(dataset, "df"):
        return dataset.df["has_hole"].tolist()
    # If WrappedDataset passed, try to access base
    if hasattr(dataset, "base"):
        return extract_has_hole_list(dataset.base)
    raise RuntimeError("Dataset does not expose labels via .data or .df")


# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """
    Train for one epoch.
    Input:
        model, loader, optimizer, criterion, device
        scaler: GradScaler or None (if AMP available)
    Output:
        avg_loss (float)
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * imgs.size(0)
        n_samples += imgs.size(0)

    return running_loss / max(1, n_samples)


def evaluate(model, loader, device):
    """
    Evaluate model on loader.
    Returns:
        acc, classification_report_str, confusion_matrix, y_true (np), y_probs (np)
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labs = labels.cpu().numpy().flatten()

            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())

    y_true = np.array(all_labels, dtype=np.int32)
    y_probs = np.array(all_probs, dtype=np.float32)
    # default threshold 0.5
    y_pred = (y_probs >= 0.5).astype(int)
    acc = (y_pred == y_true).mean()
    report = classification_report(y_true, y_pred, target_names=["No Hole", "Hole"], zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return acc, report, cm, y_true, y_probs


def tune_threshold(y_true, y_probs):
    """
    Find threshold maximizing F1 for positive class using precision-recall curve.
    Input:
        y_true (np.array), y_probs (np.array)
    Output:
        best_thr (float), best_f1 (float)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # thresholds length = len(precisions) - 1
    best_thr = 0.5
    best_f1 = -1.0
    for i, thr in enumerate(thresholds):
        p = precisions[i + 1]
        r = recalls[i + 1]
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


# -------------------------
# Main - training orchestration
# -------------------------
def main():
    # Device
    device = get_device()
    print("Device:", device)

    # DataLoader worker settings
    num_workers, pin_memory, persistent_workers = dataloader_worker_settings(device)
    print("num_workers:", num_workers, "pin_memory:", pin_memory, "persistent_workers:", persistent_workers)

    # Load base datasets (they return tensor images CHW in [0,1])
    print("Loading datasets...")
    train_base = ClassifierDataset(txt_path=TRAIN_TXT, images_root=TRAIN_ROOT, img_size=IMG_SIZE)
    val_base = ClassifierDataset(txt_path=VAL_TXT, images_root=VAL_ROOT, img_size=IMG_SIZE)
    test_base = ClassifierDataset(txt_path=TEST_TXT, images_root=TEST_ROOT, img_size=IMG_SIZE)

    # Extract class counts
    train_labels_list = extract_has_hole_list(train_base)
    num_pos = sum(train_labels_list)
    num_neg = len(train_labels_list) - num_pos
    print(f"Train size: {len(train_base)}  Pos: {num_pos}, Neg: {num_neg}")

    # Weighted sampler
    class_counts = [num_neg, num_pos]
    class_weights = [sum(class_counts) / c for c in class_counts]
    sample_weights = [class_weights[int(v)] for v in train_labels_list]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print("Making Wrapping")
    # transforms & wrapped datasets
    print("Making transformers")
    train_tfms, val_tfms = get_transforms(IMG_SIZE)
    print("Making Wrepping training")
    train_ds = WrappedDataset(train_base, transform=train_tfms)
    print("Making Wrepping val")
    val_ds = WrappedDataset(val_base, transform=val_tfms)
    test_ds = WrappedDataset(test_base, transform=val_tfms)
    print("Data Loading")
    # dataloaders
    print("Data Loading train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    print("Data Loading validation")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    print("Model")
    # Build model
    model = build_model(BACKBONE, pretrained=PRETRAINED, feature_extract=FEATURE_EXTRACT)
    model.to(device)
    print("Loss")
    # Loss
    pos_weight = torch.tensor((num_neg / max(1, num_pos)), dtype=torch.float32).to(device)
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("Optimizing")
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=LR_REDUCE_PATIENCE, verbose=True)
    print("Scaler")
    # AMP scaler only if CUDA
    scaler = GradScaler() if device.type == "cuda" else None
    print("Training")
    # training loop with early stopping & checkpointing
    best_val_f1 = -1.0
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch} starts")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        print(f"Epoch {epoch} trained")
        # Evaluate
        print(f"Epoch {epoch} evaluation")
        val_acc, val_report, val_cm, y_val_true, y_val_probs = evaluate(model, val_loader, device)
        # compute val_loss for scheduler using criterion on logits
        model.eval()
        print(f"Epoch {epoch} grads")
        with torch.no_grad():
            val_logits = []
            val_labels_arr = []
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).unsqueeze(1)
                val_logits.append(model(imgs).cpu())
                val_labels_arr.append(labels.cpu())
            if val_logits:
                val_logits = torch.cat(val_logits, dim=0)
                val_labels_arr = torch.cat(val_labels_arr, dim=0)
                val_loss = float(nn.BCEWithLogitsLoss(pos_weight=pos_weight)(val_logits, val_labels_arr))
            else:
                val_loss = float("nan")
        print(f"Epoch {epoch} step")
        scheduler.step(val_loss if not math.isnan(val_loss) else 0.0)

        # threshold tuning
        best_thr, best_thr_f1 = tune_threshold(y_val_true, y_val_probs)

        epoch_time = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} - time {epoch_time:.1f}s")
        print(f"  Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}")
        print("  Val classification report:")
        print(val_report)
        print("  Confusion matrix:\n", val_cm)
        print(f"  Best val threshold (f1): {best_thr:.3f} (f1={best_thr_f1:.3f})")
        print("-" * 60)

        # compute val f1 at best_thr and checkpoint
        val_pred_best = (y_val_probs >= best_thr).astype(int)
        val_f1_at_best = f1_score(y_val_true, val_pred_best)
        if val_f1_at_best > best_val_f1 + MIN_IMPROVEMENT:
            best_val_f1 = val_f1_at_best
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  Saved best model.")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print("Early stopping (no improvement).")
            break

    # load best model and evaluate on test set
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.to(device)

    val_acc, val_report, val_cm, y_val_true, y_val_probs = evaluate(model, val_loader, device)
    best_thr, _ = tune_threshold(y_val_true, y_val_probs)
    print("Final val report (best model):")
    print(val_report)
    print(f"Final threshold chosen: {best_thr:.3f}")

    test_acc, test_report, test_cm, y_test_true, y_test_probs = evaluate(model, test_loader, device)
    test_pred_best = (y_test_probs >= best_thr).astype(int)
    print("Test report (at threshold):")
    print(classification_report(y_test_true, test_pred_best, target_names=["No Hole", "Hole"], zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test_true, test_pred_best))

    # small sample table
    n_show = min(50, len(y_test_true))
    df_compare = pd.DataFrame({
        "Image #": list(range(1, n_show + 1)),
        "Actual": y_test_true[:n_show],
        "Prob": np.round(y_test_probs[:n_show], 3),
        "Pred": test_pred_best[:n_show],
        "Correct": (y_test_true[:n_show] == test_pred_best[:n_show])
    })
    print(df_compare.head(50))


if __name__ == "__main__":
    # Run main under guard to make DataLoader multiprocess safe on macOS
    main()
