import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, model_name="Model", split_name="Test", normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    fmt = 'd' 

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f' 

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title(f'{model_name} - {split_name} Seti Hata Matrisi\n(Normalize: {normalize})')
    plt.ylabel('Gerçek Durum')
    plt.xlabel('Modelin Tahmini')

    save_path = 'artifacts/figures/'
    os.makedirs(save_path, exist_ok=True)
    
    filename = f"cm_{split_name.lower()}_{'norm' if normalize else 'raw'}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=300) 
    plt.show()
