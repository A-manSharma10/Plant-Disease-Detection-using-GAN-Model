# 🌱 DCGAN for PlantVillage Dataset

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate realistic plant leaf images from the PlantVillage dataset. The goal is to augment datasets or explore synthetic image generation in agricultural AI.

## 🚀 Features

- **DCGAN architecture** with a Generator and Discriminator trained adversarially
- **Image augmentation pipeline** for robust training
- **Training metrics tracked**: Accuracy, Precision, Recall, F1 Score, AUROC
- **Image generation and visualization** every 5/20 epochs
- **Model checkpoints** and generated samples saved to Google Drive
- **Plots** of training loss and evaluation metrics

## 🧠 Model Architecture

- **Generator**: 4-layer transposed convolutional network
- **Discriminator**: 4-layer convolutional network with LeakyReLU
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam

## 📁 Dataset

- Source: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Format: Folder-based image classification
- Used a subset of 1000 images for faster training and experimentation

## 🖼️ Output Examples

Generated images are saved every 5 epochs and visualized every 20 epochs to monitor progress.

## 🛠️ Dependencies

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- numpy

## 📂 Folder Structure

