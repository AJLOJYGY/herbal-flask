from flask import Flask, render_template, request
import numpy as np
import os
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
IMG_SIZE = 260

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_herbal.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kelas sesuai label training
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

            img = Image.open(filepath).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict with TFLite
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            pred_class = class_names[np.argmax(output)]
            confidence = f"{np.max(output) * 100:.2f}%"

            prediction = pred_class
            filename = filepath

    return render_template('index.html', prediction=prediction, confidence=confidence, image_path=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
