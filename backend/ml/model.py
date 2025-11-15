from utils import ClassifierDataset, HolesRecognitionDataset
import pandas as pd
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim

shape = [640,640,3]

train_dataset_classifier = ClassifierDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train", img_size= shape)
val_dataset_classifier = ClassifierDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid", img_size= shape)
test_dataset_classifier = ClassifierDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test", img_size= shape)

train_loader_classifier = DataLoader(train_dataset_classifier, batch_size=8, shuffle=True)
val_loader_classifier = DataLoader(val_dataset_classifier, batch_size=8, shuffle=True)
test_loader_classifier = DataLoader(test_dataset_classifier, batch_size=8, shuffle=True)

train_dataset_recognition = HolesRecognitionDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/train", img_size= shape)
val_dataset_recognition = HolesRecognitionDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/valid", img_size= shape)
test_dataset_recognition = HolesRecognitionDataset(txt_path= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test/_annotations.txt",images_root= "backend/ml/data/Dash Cam pothole Detection.v2i.yolokeras/test", img_size= shape)

train_loader_recognition = DataLoader(train_dataset_recognition, batch_size=8, shuffle=True)
val_loader_recognition = DataLoader(val_dataset_recognition, batch_size=8, shuffle=True)
test_loader_recognition = DataLoader(test_dataset_recognition, batch_size=8, shuffle=True)

print(type(train_loader_classifier))
print(train_loader_classifier)

