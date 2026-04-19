import torch
import numpy as np
import os
import sys
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import set_project_plot_style
from src.evaluation.metrics import evaluate_model
from src.evaluation.reporter import save_results
from src.visualization.confusion_matrix import plot_confusion_matrix
from src.visualization.gallery import plot_prediction_gallery
from src.visualization.grad_cam import generate_grad_cam_visualization

set_project_plot_style()

class_names = ['Saglikli', 'Pasli_Yaprak', 'Bakteriyel_Leke']
dummy_labels = np.random.randint(0, 3, 20)
dummy_preds = np.random.randint(0, 3, 20)
dummy_confs = np.random.uniform(0.7, 0.99, 20)
dummy_imgs = [np.random.rand(224, 224, 3) for _ in range(20)]

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
input_tensor = torch.randn(1, 3, 224, 224)

print(" Test süreci başlatılıyor...\n")

rapor = evaluate_model(dummy_labels, dummy_preds, class_names)
save_results(rapor)

plot_confusion_matrix(dummy_labels, dummy_preds, class_names)

plot_prediction_gallery(dummy_imgs, dummy_labels, dummy_preds, dummy_confs, class_names)

generate_grad_cam_visualization(model, input_tensor, dummy_imgs[0], model.features[-1])

print("\n Tüm 'artifact' dosyaları başarıyla üretildi.")
