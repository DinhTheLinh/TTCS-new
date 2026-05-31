import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HardTripletDataset(Dataset):
    def __init__(self, json_path, preprocess, is_train=False):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.preprocess = preprocess
        self.is_train = is_train
        
        if self.is_train:
            self.sketch_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.RandomRotation(degrees=15),  
                transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
            ])
            self.photo_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), 
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        anchor_img = Image.open(item["anchor"]).convert("RGB")
        positive_img = Image.open(item["positive"]).convert("RGB")
        negative_img = Image.open(item["negative"]).convert("RGB")
        
        label = item["class_name"]
        
        if self.is_train:
            anchor_img = self.sketch_augment(anchor_img)
            positive_img = self.photo_augment(positive_img)
            negative_img = self.photo_augment(negative_img)
            
        anchor = self.preprocess(anchor_img)
        positive = self.preprocess(positive_img)
        negative = self.preprocess(negative_img)

        return anchor, positive, negative, label