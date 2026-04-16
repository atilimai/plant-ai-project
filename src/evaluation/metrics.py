from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(true_labels, pred_labels, class_names):
    # 1. Per-class precision, recall ve F1 hesapla
    report = classification_report(true_labels, pred_labels,
                                   target_names=class_names,
                                   output_dict=True)

    print("--- Model Değerlendirme Raporu ---")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    return report
