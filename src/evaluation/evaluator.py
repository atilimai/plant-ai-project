import torch

def run_evaluation(model, test_loader, device):
  
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probabilities, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(conf.cpu().numpy())

    return all_labels, all_preds, all_confidences
