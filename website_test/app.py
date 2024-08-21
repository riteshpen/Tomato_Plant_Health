import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Define the model architecture
def build_model():
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(256, 256, 3),
                                                      include_top=False,
                                                      weights=None)  # No pre-trained weights
    base_model.trainable = False  # Freeze the base model
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(3, activation='softmax')  # Assuming 3 classes
    ])
    return model

model = build_model()

# Load the trained weights
model.load_weights("/Users/ritesh/tomato_disease_classification/models/tomato-disease/saved_models/models/model_3.weights.h5")

# Recompile the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

# Function to process the uploaded image
def process_image(image_data):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to make predictions
def predict(image_data):
    processed_image = process_image(image_data)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit UI
st.title("Tomato Plant Disease Classifier")
st.write("Upload an image of a tomato plant to classify it as healthy, late blight, or early blight.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    prediction = predict(image)
    class_names = ['Healthy', 'Late Blight', 'Early Blight']
    predicted_class = class_names[np.argmax(prediction)]
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Raw prediction: {prediction}")
