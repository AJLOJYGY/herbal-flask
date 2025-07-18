from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model_herbal.h5')
IMG_SIZE = 260

class_names = ['Belimbing_Wuluh', 'Jambu_Biji', 'Katuk', 'Kelor', 'Kemangi', 'Kembang_Sepatu', 'Sirih', 'Sirsak']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('static', 'uploaded_image.jpg')
            file.save(filepath)

            img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img) / 255.
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            predicted_class = class_names[np.argmax(pred)]
            confidence = f"{np.max(pred) * 100:.2f}%"

            prediction = predicted_class
            filename = filepath

    return render_template('index.html', prediction=prediction, confidence=confidence, image_path=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

