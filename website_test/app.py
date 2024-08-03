import os
import traceback
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your trained model
model_path = '/Users/ritesh/tomato_disease_classification/models/tomato-disease/saved_model/models/model_1.h5'
try:
    model = tf.saved_model.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(traceback.format_exc())

# Define class names
class_names = ['Early Blight', 'Late Blight', 'Healthy']

def preprocess_image(image):
    try:
        img = Image.open(image).convert('RGB')
        print(f"Image opened successfully. Size: {img.size}")
        img = img.resize((224, 224))  # Adjust size according to your model's input shape
        print(f"Image resized to {img.size}")
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Image preprocessed. Shape: {img_array.shape}")
        return img_array
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            try:
                img_array = preprocess_image(file)
                input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
                infer = model.signatures["serving_default"]
                predictions = infer(input_tensor)
                output_tensor = predictions['dense_2'] 
                
                predicted_class = class_names[np.argmax(output_tensor.numpy()[0])]
                confidence = float(np.max(output_tensor.numpy()[0]))
                
                return jsonify({
                    'prediction': predicted_class,
                    'confidence': confidence
                })
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                print(traceback.format_exc())
                return jsonify({'error': str(e)})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)