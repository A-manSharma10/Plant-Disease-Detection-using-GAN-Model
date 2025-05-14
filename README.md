🌿 Plant Disease Detection using GANs, ResNet34 & MobileNetV2
This project implements an end-to-end plant disease detection system using Generative Adversarial Networks (DCGAN) for data augmentation and deep learning models (ResNet34 and MobileNetV2) for classification. It also includes a Gradio-based web interface and Wikipedia API integration for contextual disease information.

🚀 Features
DCGAN-based synthetic image generation for robust dataset augmentation.

ResNet34 classifier for high-accuracy disease classification.

Hugging Face MobileNetV2 model for inference on plant disease datasets.

Gradio Web App for real-time user interaction.

Wikipedia integration to display detailed information about predicted diseases.

Evaluation metrics: Accuracy, Precision, Recall, F1 Score, AUROC, Confusion Matrix.

🗂️ Project Structure
bash
Copy
Edit
📦 Plant-Disease-Detection-GAN
├── dataset/                      # PlantVillage dataset (color version)
├── generated_images/            # Output of GAN-generated samples
├── saved_models/                # Trained model weights
├── main.py                      # Main script (DCGAN, ResNet, Web UI)
├── README.md                    # You're reading it!
📦 Requirements
Install dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
Major Libraries Used:

PyTorch

torchvision

Hugging Face Transformers

Gradio

scikit-learn

matplotlib

Pillow

requests

🧠 Models Used
DCGAN: Trained to generate realistic leaf images to increase dataset diversity.

ResNet34: Fine-tuned for multi-class classification on PlantVillage dataset.

MobileNetV2 (Hugging Face): Used for prediction with external pre-trained model.

⚙️ How It Works
1. Train the DCGAN
bash
Copy
Edit
python main.py
# Select Option 1 from the menu
Generates augmented images to enhance model generalization.

2. Train ResNet34 Classifier
bash
Copy
Edit
python main.py
# Select Option 2
Fine-tunes a ResNet34 model using real + GAN-augmented data.

3. Predict from CLI
bash
Copy
Edit
python main.py
# Select Option 3
# Input path to a leaf image
Predicts disease using the pre-trained MobileNetV2 model.

4. Launch Web Interface
bash
Copy
Edit
python main.py
# Select Option 4
A user-friendly Gradio web app appears to:

Upload images

View predictions

Read detailed disease info from Wikipedia

📊 Evaluation Metrics
Accuracy, Precision, Recall, F1 Score, AUROC (plotted after training)

Confusion Matrix for real vs. fake (GAN)

GAN loss curves (Discriminator vs Generator)

## 📈 Practical Results

- 🚀 **Accuracy Boost:** GAN-based data augmentation improved classification accuracy from **75.45% to 85.60%**, and F1-score from **0.7696 to 0.8715**.
- 🧠 **Stable Training:** Discriminator loss reduced from **0.3531 to 0.1636** and AUROC peaked at **1.0**, indicating highly reliable classification performance.
- 🌿 **Live Prediction Interface:** Developed a Gradio-based web UI using **MobileNetV2** for real-time plant disease prediction and Wikipedia-based disease info fetch.


📌 Dataset
PlantVillage Dataset (color)

Place it in:

swift
Copy
Edit
/content/drive/MyDrive/kaggle_datasets/plantvillage/color
✍️ Author
Aman Sharma
LinkedIn | GitHub

📄 License
This project is licensed under the MIT License.
