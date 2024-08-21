import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def build_model():
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(256, 256, 3),
                                                      include_top=False,
                                                      weights=None)
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

st.title("Tomato Plant Disease Classifier")
st.write("Upload your model weights file and an image to classify it.")

# Upload the model weights file
uploaded_weights = st.file_uploader("Upload model weights...", type=["jpeg"])
if uploaded_weights is not None:
    model = build_model()
    try:
        model.load_weights(uploaded_weights)
        model.compile(optimizer='adam', 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                      metrics=['accuracy'])
        st.write("Model weights loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model weights: {e}")

    # Upload the image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")

        def process_image(image_data):
            size = (256, 256)
            image = ImageOps.fit(image_data, size, Image.LANCZOS)
            img = np.asarray(image)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            return img

        def predict(image_data):
            processed_image = process_image(image_data)
            prediction = model.predict(processed_image)
            return prediction

        try:
            prediction = predict(image)
            class_names = ['Healthy', 'Late Blight', 'Early Blight']
            predicted_class = class_names[np.argmax(prediction)]
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Raw prediction: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.write("Please upload the model weights file.")
