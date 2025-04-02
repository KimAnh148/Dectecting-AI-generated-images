import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import create_model, grad_cam
from torchvision import transforms

def load_and_preprocess_image(image):
    """
    Tiền xử lý ảnh đầu vào
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply transforms
    img_tensor = transform(image)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def create_heatmap(model, img_tensor, original_img):
    """
    Tạo heatmap sử dụng Grad-CAM
    """
    # Lấy heatmap
    heatmap = grad_cam(model, img_tensor)
    
    # Resize heatmap về kích thước của ảnh gốc
    heatmap = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap = torch.nn.functional.interpolate(
        heatmap, 
        size=original_img.size[::-1],
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    # Tạo bản đồ nhiệt
    plt.figure(figsize=(10, 10))
    plt.imshow(original_img)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.axis('off')
    
    # Lưu ảnh vào buffer
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    
    return buf

def main():
    st.title('Phân loại ảnh thật/giả sử dụng AI')
    
    # Tải model đã huấn luyện
    try:
        model = create_model()
        model.load_state_dict(torch.load('saved_model/best_model.pth'))
        model.eval()
    except:
        st.error('Không tìm thấy model đã huấn luyện. Vui lòng huấn luyện model trước.')
        return
    
    # Upload ảnh
    uploaded_file = st.file_uploader("Chọn một ảnh để phân loại", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Hiển thị ảnh gốc
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Ảnh gốc', use_column_width=True)
        
        # Tiền xử lý ảnh
        img_tensor = load_and_preprocess_image(image)
        
        # Dự đoán
        with torch.no_grad():
            prediction = model(img_tensor).squeeze().item()
        
        # Hiển thị kết quả
        if prediction > 0.5:
            st.success(f'Ảnh này có {prediction:.2%} khả năng là ảnh thật')
        else:
            st.error(f'Ảnh này có {1-prediction:.2%} khả năng là ảnh giả')
        
        # Tạo và hiển thị heatmap
        heatmap_buf = create_heatmap(model, img_tensor, image)
        st.image(heatmap_buf, caption='Bản đồ nhiệt Grad-CAM', use_column_width=True)

if __name__ == '__main__':
    main() 