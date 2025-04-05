import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ImageDataLoader:
    def __init__(self, data_dir, img_size=(32, 32), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_data(self, test_size=0.2):
        """
        Tải và tiền xử lý dữ liệu từ thư mục
        """
        image_paths = []
        labels = []
        
        # Tải ảnh thật
        real_dir = os.path.join(self.data_dir, 'real')
        for img_name in os.listdir(real_dir):
            img_path = os.path.join(real_dir, img_name)
            image_paths.append(img_path)
            labels.append(1)  # 1 cho ảnh thật
            
        # Tải ảnh giả
        fake_dir = os.path.join(self.data_dir, 'fake')
        for img_name in os.listdir(fake_dir):
            img_path = os.path.join(fake_dir, img_name)
            image_paths.append(img_path)
            labels.append(0)  # 0 cho ảnh giả
        
        # Chia dữ liệu thành tập train và test
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42
        )
        
        # Tạo datasets
        train_dataset = ImageDataset(train_paths, train_labels, self.transform)
        test_dataset = ImageDataset(test_paths, test_labels, self.transform)
        
        # Tạo dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader 