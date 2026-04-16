import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_grad_cam_visualization(model, input_tensor, rgb_img, target_layer, file_name="grad_cam_result"):

    cam = GradCAM(model=model, target_layers=[target_layer])
    
    targets = [ClassifierOutputTarget(np.argmax(model(input_tensor).detach().numpy()))]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    os.makedirs('artifacts/figures/', exist_ok=True)
    save_path = f'artifacts/figures/{file_name}.png'
    
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.title(f"Odaklanma Haritası: {file_name}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()
    
    print(f"Grad-CAM sonucu {save_path} adresine kaydedildi.")
    return visualization
