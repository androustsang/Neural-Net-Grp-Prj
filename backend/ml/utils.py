import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize an image to `new_shape` (height, width) using YOLO-style letterboxing.
    Preserves aspect ratio and pads the image with a constant color.

    Input:
        img (np.array): Original image in HWC format (RGB or BGR)
        new_shape (tuple): Target size (height, width), default (640,640)
        color (tuple): Padding color (R,G,B), default gray (114,114,114)

    Output:
        img_padded (np.array): Resized and padded image (HWC) in same dtype
    """
    shape = img.shape[:2] 
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * r), int(shape[0] * r)) 

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    top, bottom = int(dh), int(dh)
    left, right = int(dw), int(dw)

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return img_padded

def load_image(path, img_size=(640, 640)):
    """
    Load an image from disk safely and preprocess for model.

    Input:
        path (str): Path to the image file
        img_size (tuple): Output image size (H,W), default (640,640)

    Output:
        img (torch.Tensor): Float32 tensor, shape [3, H, W], values [0,1]
        Returns None if file not found or cannot be read
    """
    if not os.path.exists(path):
        return None  

    img = cv2.imread(path)
    if img is None:
        return None 

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = letterbox(img, img_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.tensor(img, dtype=torch.float32)

def parse_annotations(txt_path, images_root):
    """
    Parse YOLO-style annotation txt file.
    Format per line:
      image.jpg  x1,y1,x2,y2,class  x1,y1,x2,y2,class ...

    Input:
        txt_path (str): Path to YOLO annotation txt
        images_root (str): Folder where images are stored

    Output:
        List of dictionaries:
        [
            {
                "img_path": full path to image,
                "holes": list of bounding boxes [[x1,y1,x2,y2], ...],
                "has_hole": 1 if holes exist else 0
            },
            ...
        ]
    """
    rows = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_file = parts[0]

            img_path = os.path.join(images_root, img_file)

            holes = []
            for h in parts[1:]:
                x1, y1, x2, y2, cls = h.split(",")
                holes.append([int(x1), int(y1), int(x2), int(y2)])

            rows.append({
                "img_path": img_path,
                "holes": holes,
                "has_hole": 1 if len(holes) > 0 else 0
            })

    return rows


class ClassifierDataset(Dataset):
    """
    Dataset for binary classification (Hole / No Hole)

    Input:
        txt_path (str): Path to YOLO-style annotation txt
        images_root (str): Image folder
        img_size (tuple): Output image size (H,W)

    Output (__getitem__):
        img (torch.Tensor): Float32 tensor [3,H,W] normalized 0-1
        label (torch.Tensor): Float32 scalar 0/1
    """
    def __init__(self, txt_path, images_root, img_size=(640, 640)):
        self.data = parse_annotations(txt_path, images_root)
        self.img_size = img_size
        self.missing = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        img = load_image(row["img_path"], self.img_size)
        if img is None:
            self.missing.append(row["img_path"])
            img = torch.zeros((3, self.img_size[0], self.img_size[1]))

        label = torch.tensor(row["has_hole"], dtype=torch.float32)
        return img, label


class HolesRecognitionDataset(Dataset):
    """
    Dataset for bounding box regression (Hole detection)

    Input:
        txt_path (str): Path to YOLO annotation txt
        images_root (str): Image folder
        img_size (tuple): Output image size (H,W)

    Output (__getitem__):
        img (torch.Tensor): Float32 tensor [3,H,W] normalized 0-1
        bbox (torch.Tensor): Float32 tensor [x1,y1,x2,y2] (or [0,0,0,0] if no bbox)
    """
    def __init__(self, txt_path, images_root, img_size=(640, 640)):
        self.data = parse_annotations(txt_path, images_root)
        self.img_size = img_size
        self.missing = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        img = load_image(row["img_path"], self.img_size)
        if img is None:
            self.missing.append(row["img_path"])
            img = torch.zeros((3, self.img_size[0], self.img_size[1]))

        # Return first bbox or [0,0,0,0] if empty
        if row["holes"]:
            bbox = torch.tensor(row["holes"][0], dtype=torch.float32)
        else:
            bbox = torch.tensor([0, 0, 0, 0], dtype=torch.float32)

        return img, bbox
