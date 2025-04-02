import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import create_model
from data_loader import DataLoader
import matplotlib.pyplot as plt

def train_model(data_dir, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Huấn luyện mô hình phân loại ảnh thật/giả
    """
    # Kiểm tra GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tải dữ liệu
    data_loader = DataLoader(data_dir, batch_size=batch_size)
    train_loader, test_loader = data_loader.load_data()
    
    # Tạo mô hình
    model = create_model().to(device)
    
    # Định nghĩa loss function và optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Tạo thư mục để lưu model
    os.makedirs('saved_model', exist_ok=True)
    
    # Biến để theo dõi model tốt nhất
    best_accuracy = 0.0
    
    # Biến để lưu lịch sử huấn luyện
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Huấn luyện mô hình
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass và optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Tính toán accuracy
            predicted = (outputs.squeeze() > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        
        train_accuracy = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.float().to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                
                # Tính toán accuracy
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        
        val_accuracy = val_correct / val_total
        val_loss = val_loss / len(test_loader)
        
        # Lưu lịch sử
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        # In thông tin
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Lưu model tốt nhất
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'saved_model/best_model.pth')
    
    # Vẽ đồ thị accuracy và loss
    plot_training_history(history)
    
    return model

def plot_training_history(history):
    """
    Vẽ đồ thị accuracy và loss trong quá trình huấn luyện
    """
    # Vẽ accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Vẽ loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == '__main__':
    # Đường dẫn đến thư mục chứa dữ liệu
    data_dir = 'path_to_your_data_directory'  # Thay đổi đường dẫn này
    
    # Huấn luyện mô hình
    model = train_model(data_dir) 