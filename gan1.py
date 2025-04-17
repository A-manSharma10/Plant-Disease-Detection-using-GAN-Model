import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# ✅ Set Device
device = torch.device("cpu")  # Use CPU
use_amp = False

# ✅ Data Transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Load Dataset
dataset_path = "/content/drive/MyDrive/kaggle_datasets/plantvillage/color"
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
fixed_indices = np.random.choice(len(dataset), 1000, replace=False)
subset = torch.utils.data.Subset(dataset, fixed_indices)
dataloader = torch.utils.data.DataLoader(subset, batch_size=8, shuffle=True, pin_memory=False)

# ✅ Output Path
output_path = "/content/drive/MyDrive/generated_images/"
model_path = "/content/drive/MyDrive/saved_models/"
os.makedirs(output_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# ✅ DCGAN Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x.view(-1, 100, 1, 1))

# ✅ DCGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)

# ✅ Initialize Models
G = Generator().to(device)
D = Discriminator().to(device)

# ✅ Optimizer Setup
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))

# ✅ Training Variables
num_epochs = 100
fixed_noise = torch.randn(16, 100, 1, 1).to(device)

# ✅ Metrics Tracking
train_losses_G, train_losses_D = [], []
accuracies, f1_scores, precisions, recalls, aurocs = [], [], [], [], []

# ✅ Training Loop
for epoch in range(num_epochs):
    all_labels, all_preds = [], []
    total_d_loss, total_g_loss = 0, 0
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        d_loss_real = criterion(D(real_images), real_labels)
        z = torch.randn(batch_size, 100, 1, 1).to(device)
        fake_images = G(z)
        d_loss_fake = criterion(D(fake_images.detach()), fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator More Frequently
        for _ in range(2):
            optimizer_G.zero_grad()
            fake_images = G(z)
            g_loss = criterion(D(fake_images), real_labels)
            g_loss.backward()
            optimizer_G.step()

        # Track losses
        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()

        # Collect Metrics
        real_preds = D(real_images).detach().cpu().numpy().flatten().tolist()
        fake_preds = D(fake_images.detach()).detach().cpu().numpy().flatten().tolist()
        all_labels.extend([1] * batch_size + [0] * batch_size)
        all_preds.extend(real_preds + fake_preds)

    # ✅ Calculate Metrics
    binary_preds = [1 if i > 0.5 else 0 for i in all_preds]
    accuracy = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    auroc = roc_auc_score(all_labels, all_preds)

    accuracies.append(accuracy)
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)
    aurocs.append(auroc)
    train_losses_G.append(total_g_loss / len(dataloader))
    train_losses_D.append(total_d_loss / len(dataloader))

    # ✅ Print Epoch Results
    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {total_d_loss:.4f} | G Loss: {total_g_loss:.4f} | "
          f"Acc: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUROC: {auroc:.4f}")

    # ✅ Save Generated Images every 5 epochs
    if (epoch + 1) % 5 == 0:
        fake_images = G(fixed_noise).detach().cpu()
        save_image(fake_images, os.path.join(output_path, f"generated_epoch_{epoch+1}.png"), normalize=True)

    # ✅ Show Generated Images Every 20 Epochs
    if (epoch + 1) % 20 == 0:
        fake_images = G(fixed_noise).detach().cpu()
        grid = torchvision.utils.make_grid(fake_images, normalize=True)
        plt.figure(figsize=(6,6))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()


# ✅ Save Model Weights (Learning Save!)
torch.save(G.state_dict(), os.path.join(model_path, "generator.pth"))
torch.save(D.state_dict(), os.path.join(model_path, "discriminator.pth"))

# ✅ End Training Timer
end_time = time.time()
training_time = end_time - start_time
print(f"Total Training Time: {training_time:.2f} seconds")

# ✅ Plot Losses
plt.figure(figsize=(10,5))
plt.plot(train_losses_G, label='Generator Loss')
plt.plot(train_losses_D, label='Discriminator Loss')
plt.legend()
plt.show()

# ✅ Plot Metrics
plt.figure(figsize=(12,6))
plt.plot(accuracies, label="Accuracy", marker="o")
plt.plot(precisions, label="Precision", marker="s")
plt.plot(recalls, label="Recall", marker="^")
plt.plot(f1_scores, label="F1 Score", marker="D")
plt.plot(aurocs, label="AUROC", marker="*")
plt.legend()
plt.show()


