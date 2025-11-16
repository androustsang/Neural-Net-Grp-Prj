from model import CNNClassifier
from utils import ClassifierDataset, HolesRecognitionDataset
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

shape = [224,224]

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


def evaluate():
    model.eval()
    total_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader_classifier:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_correct += (preds == labels).sum().item()
            total += labels.numel()

    return total_correct / total


if __name__ == "__main__":
    train_dataset_classifier = ClassifierDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train", img_size= shape)
    val_dataset_classifier = ClassifierDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid", img_size= shape)

    train_loader_classifier = DataLoader(train_dataset_classifier, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_classifier = DataLoader(val_dataset_classifier, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)


    train_dataset_recognition = HolesRecognitionDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train", img_size= shape)
    val_dataset_recognition = HolesRecognitionDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid", img_size= shape)
    test_dataset_recognition = HolesRecognitionDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test", img_size= shape)

    train_loader_recognition = DataLoader(train_dataset_recognition, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_recognition = DataLoader(val_dataset_recognition, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    test_loader_recognition = DataLoader(test_dataset_recognition, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNNClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)
    
    labels_list = [label.item() for _, label in train_dataset_classifier]
    num_pos = sum(labels_list)              
    num_neg = len(labels_list) - num_pos 
    
    epochs = 5
    for epoch in range(epochs):
        train_loss = train_one_epoch()
        acc = evaluate()

        print(f"Epoch {epoch+1}/{epochs} "
            f"- Loss: {train_loss:.4f} "
            f"- Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), "backend/ml/models/binary_classifier.pth")
    print("Model saved.")