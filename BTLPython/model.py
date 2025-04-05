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

    activations = None
    gradients = None

    # Hàm hook để lấy đầu ra của conv3
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output  # Lưu đầu ra của conv3

    # Hàm hook để lấy gradient của conv3
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]  # Lưu gradient của conv3

    # Đăng ký hook cho conv3
    forward_handle = model.conv3.register_forward_hook(forward_hook)
    backward_handle = model.conv3.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)

    # Backward pass cho lớp target_class
    model.zero_grad()
    output.backward(torch.ones_like(output))

    # Xóa hook sau khi lấy dữ liệu
    forward_handle.remove()
    backward_handle.remove()

    # Kiểm tra nếu gradients không None
    if gradients is None or activations is None:
        raise ValueError("Không lấy được gradients hoặc activations!")

    # Global average pooling of gradients
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

    # Tính Grad-CAM
    cam = torch.sum(weights * activations, dim=1, keepdim=True)

    # Apply ReLU và chuẩn hóa
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-9)  # Tránh chia cho 0

    return cam.detach().cpu().numpy()[0, 0]
