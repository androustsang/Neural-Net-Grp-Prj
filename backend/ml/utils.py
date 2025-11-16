import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def parse_annotations(txt_path, images_root):
    rows = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            img_path = f"{images_root}/{img_name}"

            holes = []
            for h in parts[1:]:
                x1, y1, x2, y2, cls = h.split(",")
                holes.append([int(x1), int(y1), int(x2), int(y2)])

            rows.append({
                "img_path": img_path,
                "holes": holes,
                "has_hole": 1 if len(holes) > 0 else 0
            })

    return pd.DataFrame(rows)

# df_train = parse_annotations("backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train/_annotations.txt", "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train")
# df_val = parse_annotations("backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid/_annotations.txt", "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid")
# df_test = parse_annotations("backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test/_annotations.txt", "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test")


class ClassifierDataset(Dataset):
    def __init__(self, txt_path, images_root, img_size=(640, 640)):
        self.df = parse_annotations(txt_path, images_root).reset_index(drop=True)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img = cv2.imread(row["img_path"])
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)

        label = torch.tensor(row["has_hole"], dtype=torch.float32)

        return img, label

class HolesRecognitionDataset(Dataset):
    """
    Dataset for object recognition (bounding box regression)
    """
    def __init__(self, txt_path, images_root, img_size=(640, 640)):
        self.df = parse_annotations(txt_path, images_root).reset_index(drop=True)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(row["img_path"])
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = torch.tensor(img, dtype=torch.float32)

        if len(row["holes"]) > 0:
            holes = torch.tensor(row["holes"][0], dtype=torch.float32)
        else:
            holes = torch.tensor([0, 0, 0, 0], dtype=torch.float32)

        return img, holes

