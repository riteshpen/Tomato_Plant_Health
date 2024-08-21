import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import io

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

# Streamlit UI
st.title("Tomato Plant Disease Classifier")
st.write("Upload your model weights file and an image to classify it.")

# Upload the model weights file
uploaded_weights = st.file_uploader("Upload model weights...", type=["jpeg"])
if uploaded_weights is not None:
    model = build_model()
    model.load_weights(uploaded_weights)
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=['accuracy'])
    st.write("Model weights loaded successfully.")

    # Upload the image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")

        # Predict
        def process_image(image_data):
            size = (256, 256)  # Input size used during training
            image = ImageOps.fit(image_data, size, Image.LANCZOS)
            img = np.asarray(image)
            img = img / 255.0  # Rescale the image
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img

        def predict(image_data):
            processed_image = process_image(image_data)
            prediction = model.predict(processed_image)
            return prediction

        prediction = predict(image)
        class_names = ['Healthy', 'Late Blight', 'Early Blight']  # Update based on your model's classes
        predicted_class = class_names[np.argmax(prediction)]

        st.write(f"Prediction: {predicted_class}")
        st.write(f"Raw prediction: {prediction}")
