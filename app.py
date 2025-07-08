from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle

app = Flask(__name__)

# Load the trained model
model_path='model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file.filename == '':
        return 'No file selected'

    # Read the image in-memory without saving
    image = Image.open(file.stream).convert('RGB')
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.reshape(image, (1, 128, 128, 3))

    # Predict
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    result = "Wearing a Mask 😷" if predicted_class == 1 else "Not Wearing a Mask 😐"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
