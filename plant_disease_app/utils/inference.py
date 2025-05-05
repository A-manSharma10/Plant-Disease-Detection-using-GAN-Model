%%writefile /content/plant_disease_app/utils/inference.py
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained MobileNetV2 model
def predict(image: Image.Image):
    model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = logits.argmax(-1).item()
        class_name = model.config.id2label.get(predicted_class, "Unknown")

    return class_name

# Function to fetch detailed info from Wikipedia
import requests
from urllib.parse import quote

def get_detailed_wikipedia_info(query):
    try:
        # Search for the article
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(query)}&format=json"
        search_response = requests.get(search_url).json()
        search_results = search_response.get("query", {}).get("search", [])
        
        if not search_results:
            return query, "No Wikipedia pages found."

        # Get full page content
        top_title = search_results[0]["title"]
        extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=&explaintext=&titles={quote(top_title)}&format=json"
        extract_response = requests.get(extract_url).json()

        page_data = extract_response.get("query", {}).get("pages", {})
        page = next(iter(page_data.values()))
        extract_text = page.get("extract", "No detailed info found.")

        page_link = f"https://en.wikipedia.org/wiki/{quote(top_title.replace(' ', '_'))}"
        info = (
            f"### 🌿 **{top_title}**\n\n"
            f"{extract_text.strip()}\n\n"
            f"🔗 [Click here to read more on Wikipedia]({page_link})"
        )

        return top_title, info

    except Exception as e:
        return query, f"Error fetching info: {str(e)}"
