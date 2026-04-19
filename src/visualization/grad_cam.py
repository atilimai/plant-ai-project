import os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_grad_cam_visualization(model, input_tensor, rgb_img, target_layer, file_name="grad_cam_result"):
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(np.argmax(model(input_tensor).detach().numpy()))]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title("Orijinal Resim", fontsize=10)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(grayscale_cam, cmap='jet')
    plt.title("Isı Haritası (Heatmap)", fontsize=10)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(visualization)
    plt.title("Overlay Blend", fontsize=10)
    plt.axis('off')
    
    os.makedirs('artifacts/figures/', exist_ok=True)
    save_path = f'artifacts/figures/{file_name}_triple.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return visualization
