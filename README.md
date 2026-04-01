# 🫁 Pneumonia Detection from Chest X-Rays

A deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using CNN (from scratch) and ResNet18 (transfer learning), with Grad-CAM visualizations for model interpretability.

## 📌 Project Overview

Pneumonia is a serious lung infection detectable through chest X-rays. This project builds and compares two AI models to assist in automated diagnosis, achieving up to **81% accuracy** with the custom CNN.

## 📁 Dataset

- **Name:** Chest X-Ray Images (Pneumonia) — Kermany et al.
- **Source:** [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** ~5,000 labeled images (NORMAL / PNEUMONIA)

## 🏗️ Models

| Model | Accuracy | Precision (Pneumonia) | Recall (Pneumonia) |
|---|---|---|---|
| CNN (from scratch) | 81% | 0.77 | 0.98 |
| ResNet18 (Transfer Learning) | 79% | 0.76 | 0.98 |

## ⚙️ Pipeline

1. Load & visualize X-ray samples
2. Resize, normalize, and augment images
3. Train CNN from scratch (2 conv layers, ReLU, MaxPool, FC)
4. Fine-tune ResNet18 pretrained on ImageNet
5. Evaluate using accuracy, precision, recall, F1-score, confusion matrix
6. Apply **Grad-CAM** for visual interpretability

## 🔍 Grad-CAM Visualization

Grad-CAM highlights the regions of the X-ray the model focused on when making predictions, helping explain model decisions visually.

## 🚀 How to Run
```bash
pip install torch torchvision matplotlib numpy
jupyter notebook i233022_Faryal.ipynb
```

## 📊 Key Results

- CNN achieved higher precision than ResNet18
- ResNet18 trained faster but was slightly less precise
- Grad-CAM confirmed the model focuses on lung regions

## 👩‍💻 Author

**Faryal Siddique** — 23i3022  
BS Artificial Intelligence, FAST-NUCES Islamabad
