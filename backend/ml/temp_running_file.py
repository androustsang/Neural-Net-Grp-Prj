from model import CNNClassifier
from utils import ClassifierDataset, HolesRecognitionDataset
from torch.utils.data import DataLoader
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

shape = [224,224]

pd.set_option('display.max_rows', None)

test_dataset_classifier = ClassifierDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test", img_size= shape)
test_loader_classifier = DataLoader(test_dataset_classifier, batch_size=8, shuffle=True)

model = CNNClassifier()
model.load_state_dict(torch.load("backend/ml/models/binary_classifier.pth", map_location=device))
model.to(device)
model.eval()

actual_labels = []
pred_labels = []

total_correct = 0
total_samples = 0

with torch.no_grad():
    for i, (img, label) in enumerate(test_loader_classifier):
        img = img.to(device)
        label = label.to(device).unsqueeze(1)

        logits = model(img)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()

        # Count overall accuracy
        total_correct += (pred == label).sum().item()
        total_samples += label.size(0)

        # For first 50 images, save readable labels
        if i < 50:
            actual_labels.extend(label.cpu().numpy().flatten().tolist())
            pred_labels.extend(pred.cpu().numpy().flatten().tolist())

# Overall test accuracy
accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy:.4f}")

# Presentable comparison for first 50 images
df_comparison = pd.DataFrame({
    "Image #": list(range(1, len(actual_labels)+1)),
    "Actual Label": actual_labels,
    "Predicted Label": pred_labels,
    "Correct": [a==p for a,p in zip(actual_labels, pred_labels)]
})

print(df_comparison)


