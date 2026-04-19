# Model Artifacts & Management

This directory serves as the centralized repository for trained model weights and architecture metadata for the Plant Disease Classification project.

## 📂 Directory Structure
- **`/checkpoints`**: Contains intermediate PyTorch state dictionaries (`.pth`) saved during training epochs for recovery and analysis.
- **`/exported`**: Stores final, production-ready models optimized for inference (Binary #5 and Multiclass #6 tracks).

## ⚖️ Storage Policy
- **Format:** All models must be saved in PyTorch `state_dict` format.
- **Naming Convention:** `[task]_[arch]_v[x].pth` (e.g., `binary_mobilenetv2_v1.pth`).
- **Metadata:** Each final model should include a version log detailing the training parameters (LR, Epochs, Optimizer).

> **Note:** Large binary files are managed via Git LFS or external cloud references to maintain repository performance.
