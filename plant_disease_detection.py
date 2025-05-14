import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import gradio as gr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ========== Configuration ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/content/drive/MyDrive/kaggle_datasets/plantvillage/color"
output_path = "/content/drive/MyDrive/generated_images/"
model_path = "/content/drive/MyDrive/saved_models/"

os.makedirs(output_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

subset_indices = np.random.choice(len(dataset), 1000, replace=False)
subset = torch.utils.data.Subset(dataset, subset_indices)
dataloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

# ========== GAN Modules ==========

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

# ========== Save GAN Models ==========

def save_gan_models(generator, discriminator):
    torch.save(generator.state_dict(), model_path + 'generator.pth')
    torch.save(discriminator.state_dict(), model_path + 'discriminator.pth')
    print("DCGAN models saved!")

# ========== Train DCGAN ==========

def train_dcgan(generator, discriminator, dataloader, num_epochs=100, lr=0.0002, device='cpu'):
    print(f"Using {device} for training...")
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    real_label = 1
    fake_label = 0

    d_loss_epoch = []
    g_loss_epoch = []

    accs, f1s, precisions, recalls, aucs = [], [], [], [], []

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        d_losses, g_losses = [], []
        all_preds = []
        all_targets = []

        for data in dataloader:
            real_data = data[0].to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, device=device).float()

            # Discriminator: real
            discriminator.zero_grad()
            output_real = discriminator(real_data).view(-1)
            err_d_real = criterion(output_real, label)
            err_d_real.backward()

            # Discriminator: fake
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_data = generator(noise)
            label.fill_(fake_label).float()
            output_fake = discriminator(fake_data.detach()).view(-1)
            err_d_fake = criterion(output_fake, label)
            err_d_fake.backward()
            optimizer_d.step()

            # Generator
            generator.zero_grad()
            label.fill_(real_label)
            output_gen = discriminator(fake_data).view(-1)
            err_g = criterion(output_gen, label)
            err_g.backward()
            optimizer_g.step()

            d_losses.append(err_d_real.item() + err_d_fake.item())
            g_losses.append(err_g.item())

            pred = torch.cat((output_real, output_fake)).detach().cpu().numpy()
            targ = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])
            all_preds.extend(pred)
            all_targets.extend(targ)

        d_epoch_loss = np.mean(d_losses)
        g_epoch_loss = np.mean(g_losses)
        bin_preds = np.array(all_preds) > 0.5
        acc = accuracy_score(all_targets, bin_preds)
        f1 = f1_score(all_targets, bin_preds)
        prec = precision_score(all_targets, bin_preds)
        rec = recall_score(all_targets, bin_preds)
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.0

        d_loss_epoch.append(d_epoch_loss)
        g_loss_epoch.append(g_epoch_loss)
        accs.append(acc)
        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        aucs.append(auc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_epoch_loss:.4f} | G Loss: {g_epoch_loss:.4f} | "
              f"Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUROC: {auc:.4f}")

        save_gan_models(generator, discriminator)

    print("Training Complete. Models saved.")

    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_epoch, label="Discriminator Loss")
    plt.plot(g_loss_epoch, label="Generator Loss")
    plt.title("GAN Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    metric_names = ["Accuracy", "F1 Score", "Precision", "Recall", "AUROC"]
    metric_values = [accs[-1], f1s[-1], precisions[-1], recalls[-1], aucs[-1]]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metric_names, metric_values, color="skyblue")
    plt.ylim(0, 1.1)
    plt.title("Final GAN Discriminator Metrics")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
    plt.grid(axis='y')
    plt.show()

    cm = confusion_matrix(all_targets, bin_preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Fake", "Real"])
    plt.yticks([0, 1], ["Fake", "Real"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ========= Train and Tune ResNet34 =============

def train_resnet34_model(dataloader, num_epochs=5):
    print("Initializing ResNet34 model...")
    resnet = models.resnet34(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)

    print("Model initialized successfully.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)

    resnet.train()

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}...")
        total_loss, correct, total = 0, 0, 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

    torch.save(resnet.state_dict(), model_path + "resnet34_classifier.pth")
    print(" ResNet34 model trained and saved.")



# ========== Predict using MobileNetV2 from Hugging Face ==========

def predict(image):
    processor = AutoImageProcessor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
    model = AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification").to(device)
    model.eval()

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = class_names[predicted_class_idx]
    return predicted_class

# ========== Web Interface ==========
from urllib.parse import quote
import requests

def get_detailed_wikipedia_info(query):
    try:
        # Step 1: Search for the article
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(query)}&format=json"
        search_response = requests.get(search_url).json()
        search_results = search_response.get("query", {}).get("search", [])

        if not search_results:
            return query, "No Wikipedia pages found."

        # Get the top result title
        top_title = search_results[0]["title"]

        # Step 2: Get full page content (plaintext extract)
        extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=&explaintext=&titles={quote(top_title)}&format=json"
        extract_response = requests.get(extract_url).json()

        page_data = extract_response.get("query", {}).get("pages", {})
        page = next(iter(page_data.values()))
        extract_text = page.get("extract", "No detailed info found.")

        # Step 3: Format output with Markdown link
        page_link = f"https://en.wikipedia.org/wiki/{quote(top_title.replace(' ', '_'))}"
        info = (
            f"### 🌿 **{top_title}**\n\n"
            f"{extract_text.strip()}\n\n"
            f"🔗 [Click here to read more on Wikipedia]({page_link})"
        )

        return top_title, info

    except Exception as e:
        return query, f"Error fetching info: {str(e)}"


# Wrapped predict function
def wrapped_predict(image):
    disease_name = predict(image)
    disease_api_name, disease_info = get_detailed_wikipedia_info(disease_name)
    return disease_name, disease_info

def launch_web():
    interface = gr.Interface(
        fn=wrapped_predict,
        inputs=gr.Image(type="pil", label="Upload Leaf Image"),
        outputs=[
            gr.Label(label="Predicted Disease"),
            gr.Textbox(label="Disease Information")
        ],
        title="Plant Disease Detection",
        description="Upload a leaf image to detect the disease."
    )
    interface.launch(share=True, debug=True)


# ========== Menu ==========

def menu():
    while True:
        print("\n====== MENU ======")
        print("1. Train DCGAN")
        print("2. Train ResNet34 Classifier")
        print("3. Predict Plant Disease")
        print("4. Launch Web Interface")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            generator = Generator().to(device)
            discriminator = Discriminator().to(device)
            train_dcgan(generator, discriminator, dataloader, num_epochs=5)
        elif choice == '2':
            train_resnet34_model(dataloader, num_epochs=5)
        elif choice == '3':
            image_path = input("Enter image path: ")
            if os.path.exists(image_path):
                result = predict(image_path)
                print(f"Prediction: {result}")
            else:
                print("Invalid image path.")
        elif choice == '4':
            launch_web()
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")

# ========== Run ==========

if __name__ == "__main__":
    menu()
