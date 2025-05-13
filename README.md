# 🧠 Fine-Tuning Pretrained CNNs on Caltech-101

This repository provides an implementation for fine-tuning convolutional neural networks (CNNs) pretrained on **ImageNet** to classify images from the **Caltech-101** dataset. It also includes a comparison with training from scratch to analyze the effectiveness of transfer learning.

---

## ✍️ Author

**Ruihan Wu**  
Fudan University  
Email: [22307140084@m.fudan.edu.cn](mailto:22307140084@m.fudan.edu.cn)

---

## 📁 Project Overview

- **Goal**: Image classification using ResNet18 on Caltech-101
- **Approach**:
  - Fine-tune a ResNet18 pretrained on ImageNet
  - Compare with a model trained from scratch
  - Perform hyperparameter search for optimal fine-tuning
- **Dataset**: [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) (101 object categories)

---

## ⚙️ Environment & Dependencies

This project supports running on:
- **Cloud platforms** (e.g., Alibaba Cloud)
- **Local machines** (with GPU support)
- **Other environments** (e.g., Google Colab, HPC clusters)

Make sure the following Python packages are installed:
- `torch`
- `torchvision`
- `matplotlib`
- `tensorboard`

---

## 📦 Project Structure

```plaintext
📁 data_preprocessing/
 ┗ 📜 data_preprocess.py        # Preprocess and split dataset

📁 training/
 ┣ 📜 pretrain.py               # Fine-tune pretrained ResNet18
 ┣ 📜 train.py                  # Train ResNet18 from scratch
 ┗ 📜 pretrain_finetuning.ipynb # Hyperparameter tuning experiments

📁 results/
 ┗ 📂 visualizations/           # Accuracy/loss curves and evaluation figures

📜 README.md                    # Project documentation

---

## 🧹 Step 1: Dataset Preparation

1. Download the dataset from [Caltech-101 Dataset](https://data.caltech.edu/records/mzrjq-6wc02).
2. Extract it to a local directory or upload it to your server.
3. Modify the dataset path in `data_preprocess.py` to match your environment.
4. Run `data_preprocess.py` to:
   - Apply data augmentation (e.g., resizing, normalization)
   - Standardize image input formats
   - Split the dataset into training, validation, and test sets
   - Ensure reproducibility using a fixed random seed

---

## 🏋️ Step 2: Training the Model

### ✅ A. Using Pretrained ResNet18

- Edit `data_dir` in `pretrain.py` to the correct dataset path.
- Run the script to:
  - Fine-tune a ResNet18 pretrained on ImageNet
  - Output training and validation accuracy per epoch
  - Evaluate final performance on the test set
  - Generate training loss and accuracy plots

> TensorBoard logs are automatically saved in the `runs/` directory for visualization.

---

### 🔧 B. Fine-Tuning Experiments

- Open and execute `pretrain_finetuning.ipynb`.
- Test different hyperparameter combinations (e.g., learning rate, layer freezing).
- Track model performance with:
  - Accuracy tables
  - Top-1 and Top-5 accuracy visualizations
- This process may take time depending on the number of combinations and hardware.

---

### ❌ C. Training from Scratch (Without Pretraining)

- Open `train.py` and set the correct `data_dir`.
- This script trains ResNet18 from randomly initialized weights using only Caltech-101.
- After training:
  - Evaluate final test accuracy
  - Compare results to pretrained models to highlight the effect of transfer learning

---

## 📈 Results & Evaluation

- Pretrained ResNet18 achieves significantly higher accuracy on Caltech-101 than models trained from scratch.
- Evaluation includes:
  - Epoch-wise loss and accuracy curves
  - Top-k accuracy plots
  - All visualizations saved in `results/visualizations/`

> Use TensorBoard or matplotlib plots to inspect learning dynamics.

---

## 🧠 Summary

This project demonstrates the power of transfer learning by fine-tuning a CNN pretrained on a large dataset (ImageNet) to achieve high accuracy on a smaller dataset (Caltech-101). With clearly structured experiments and visualizations, it serves as a strong foundation for future image classification projects.

---
