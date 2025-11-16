import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from model import CNNClassifier
from utils import ClassifierDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

shape = [224, 224]

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

num_pos = sum(train_dataset.df['has_hole'])
num_neg = len(train_dataset) - num_pos
class_counts = [num_neg, num_pos]
class_weights = [sum(class_counts)/c for c in class_counts]
sample_weights = [class_weights[int(row['has_hole'])] for idx, row in train_dataset.df.iterrows()]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = CNNClassifier().to(device)

model.load_state_dict(torch.load("backend/ml/models/binary_classifier_weighted.pth", map_location=device))

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
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

    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels_np = np.array(all_labels).flatten()
    all_preds_np = np.array(all_preds).flatten()
    val_acc = (all_preds_np == all_labels_np).mean()

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels_np, all_preds_np, target_names=["No Hole", "Hole"]))
    print("-"*60)

torch.save(model.state_dict(), "backend/ml/models/binary_classifier_weighted.pth")
print("Updated model saved.")

model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

all_labels_np = np.array(all_labels).flatten()
all_preds_np = np.array(all_preds).flatten()
test_acc = (all_preds_np == all_labels_np).mean()

print(f"Test Accuracy: {test_acc:.4f}")
print(classification_report(all_labels_np, all_preds_np, target_names=["No Hole", "Hole"]))


test_labels = all_labels_np[:50]
test_preds = all_preds_np[:50]

df_comparison = pd.DataFrame({
    "Image #": list(range(1, len(test_labels)+1)),
    "Actual Label": test_labels,
    "Predicted Label": test_preds,
    "Correct": (test_labels == test_preds)
})
print(df_comparison)

