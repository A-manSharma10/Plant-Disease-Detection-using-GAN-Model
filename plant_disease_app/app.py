
from flask import Flask, render_template, request
from PIL import Image
import os
from utils.inference import predict, get_detailed_wikipedia_info

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = info = None
    image_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        
        file = request.files['image']
        if file.filename == '':
            return "No selected file"

        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)
            image = Image.open(image_path).convert("RGB")

            prediction = predict(image)
            _, info = get_detailed_wikipedia_info(prediction)

    return render_template('index.html', prediction=prediction, info=info, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
