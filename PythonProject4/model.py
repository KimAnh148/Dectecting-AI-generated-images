import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Sigmoid activation for binary classification
        x = torch.sigmoid(x)
        
        return x

def create_model():
    """
    Tạo mô hình CNN để phân loại ảnh thật/giả
    """
    model = CNN()
    return model

def grad_cam(model, img_tensor, target_class=0):
    """
    Tạo Grad-CAM heatmap cho ảnh đầu vào
    """
    model.eval()
    img_tensor.requires_grad_()
    
    # Forward pass
    output = model(img_tensor)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass for target class
    output[0, target_class].backward()
    
    # Get gradients and activations from last conv layer
    gradients = img_tensor.grad
    activations = model.conv3.output  # Get activations from last conv layer
    
    # Global average pooling of gradients
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # Weight the channels by corresponding gradients
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    
    # Apply ReLU and normalize
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    
    return cam.detach().cpu().numpy()[0, 0] 