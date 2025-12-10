# Maaz Bobat, Saaram Rashidi, MD Sazid, Sun Hung Tsang, Yehor Valesiuk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
from model import CNNClassifier
from utils import ClassifierDataset
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
shape = [224, 224]

test_dataset_classifier = ClassifierDataset(
    txt_path="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test/_annotations.txt",
    images_root="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test",
    img_size=shape,
)
test_loader_classifier = DataLoader(
    test_dataset_classifier, batch_size=8, shuffle=False
)

train_dataset_classifier = ClassifierDataset(
    txt_path="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train/_annotations.txt",
    images_root="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train",
    img_size=shape,
)
val_dataset_classifier = ClassifierDataset(
    txt_path="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid/_annotations.txt",
    images_root="backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid",
    img_size=shape,
)

train_loader_classifier = DataLoader(
    train_dataset_classifier, batch_size=8, shuffle=True
)
val_loader_classifier = DataLoader(val_dataset_classifier, batch_size=8, shuffle=True)

model = CNNClassifier()
model.load_state_dict(
    torch.load("backend/ml/models/binary_classifier.pth", map_location=device)
)
model.to(device)
model.train()

labels_list = [label.item() for _, label in train_dataset_classifier]
num_pos = sum(labels_list)
num_neg = len(labels_list) - num_pos

pos_weight = torch.tensor(num_neg / num_pos).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)


def train_one_epoch():
    model.train()
    total_loss = 0

    for imgs, labels in train_loader_classifier:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader_classifier)


def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = [p[0] for p in all_preds]
    all_labels = [l[0] for l in all_labels]

    accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)

    report = classification_report(
        all_labels, all_preds, target_names=["No Hole", "Hole"]
    )

    return accuracy, report, all_labels, all_preds


epochs = 10
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_acc, val_report, _, _ = evaluate(val_loader_classifier)

    print(
        f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}"
    )
    print(f"Classification Report:\n{val_report}")
    print("-" * 60)

test_acc, test_report, all_labels, all_preds = evaluate(test_loader_classifier)
print(f"Test Accuracy: {test_acc:.4f}")
print(test_report)

comparison = []
for i in range(min(50, len(all_labels))):
    comparison.append(
        {
            "Image #": i + 1,
            "Actual": all_labels[i],
            "Predicted": all_preds[i],
            "Correct": all_labels[i] == all_preds[i],
        }
    )
df_comparison = pd.DataFrame(comparison)
print(df_comparison)

torch.save(model.state_dict(), "backend/ml/models/binary_classifier_weighted.pth")
print("Weighted model saved.")
