import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

from model import CNNClassifier
from utils import ClassifierDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
shape = [640, 640]
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 10

print("Doing preprocessing")

train_dataset = ClassifierDataset(
    txt_path="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train/_annotations.txt",
    images_root="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train",
    img_size=shape
)

val_dataset = ClassifierDataset(
    txt_path="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid/_annotations.txt",
    images_root="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid",
    img_size=shape
)

test_dataset = ClassifierDataset(
    txt_path="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test/_annotations.txt",
    images_root="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test",
    img_size=shape
)



num_pos = sum([item["has_hole"] for item in train_dataset.data])
num_neg = len(train_dataset) - num_pos


class_counts = [num_neg, num_pos]
class_weights = [sum(class_counts) / c for c in class_counts]

sample_weights = [class_weights[int(item["has_hole"])] for item in train_dataset.data]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Finished preprocessing")

model = CNNClassifier().to(device)

pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.NAdam(model.parameters(), lr=LR)


def train_one_epoch():
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    y_true = np.array(all_labels).flatten()
    y_pred = np.array(all_preds).flatten()

    acc = (y_true == y_pred).mean()
    report = classification_report(y_true, y_pred, target_names=["No Hole", "Hole"])

    return acc, report, y_true, y_pred


for epoch in range(EPOCHS):
    print(f"Start training {epoch+1}")
    train_loss = train_one_epoch()
    val_acc, val_report, _, _ = evaluate(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")
    print(val_report)
    print("-" * 60)

torch.save(model.state_dict(), "backend/ml/models/binary_classifier_weighted_final.pth")
print("Final weighted model saved.")

test_acc, test_report, y_true_test, y_pred_test = evaluate(test_loader)

print(f"TEST ACCURACY: {test_acc:.4f}")
print(test_report)

df_compare = pd.DataFrame({
    "Image #": list(range(1, len(y_true_test[:50]) + 1)),
    "Actual": y_true_test[:50],
    "Predicted": y_pred_test[:50],
    "Correct": (y_true_test[:50] == y_pred_test[:50])
})
print(df_compare)
