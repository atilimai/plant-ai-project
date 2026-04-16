import matplotlib.pyplot as plt
import numpy as np
import os

def plot_prediction_gallery(images, true_labels, pred_labels, confidences, class_names, n_rows=4, n_cols=5):

    n_images = n_rows * n_cols
    plt.figure(figsize=(20, 16))

    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)

        if len(images) > i:
            plt.imshow(images[i])
        else:
            plt.imshow(np.ones((224, 224, 3)) * 0.8)

        is_correct = pred_labels[i] == true_labels[i]
        color = 'green' if is_correct else 'red'

        plt.title(f"Pred: {class_names[pred_labels[i]]}\nTrue: {class_names[true_labels[i]]}\nConf: {confidences[i]:.2f}",
                  color=color, fontsize=10)

        plt.axis('off')

    plt.tight_layout()

    save_dir = 'artifacts/sample_outputs/'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'prediction_gallery.png'), dpi=300)
    plt.show()
