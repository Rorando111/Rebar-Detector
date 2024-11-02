import streamlit as st
import numpy as np
import cv2
import os
import pickle
from PIL import Image

# Load the KNN model using pickle
model_path = 'apps/OC-KNN/one_class_knn_model (1).pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Function to process and predict
def process_and_predict(img, model):
    image = np.array(img)
    if image is not None:
        # Resize the image directly for prediction (if needed)
        image = cv2.resize(image, (128, 64))
        features = image.flatten().reshape(1, -1)  # Flatten image for prediction
        distances, _ = model.kneighbors(features)
        avg_distance = np.mean(distances)

        # Simple logic to classify based on distance
        return "rebar" if avg_distance < 0.5 else "non-rebar (outlier)"
    return "unknown"

# Main function for the Streamlit app
def run():
    st.title("Rebar Classification System")

    img_file = st.file_uploader("Upload an Image for Classification", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, use_column_width=True)

        if st.button("Predict"):
            result = process_and_predict(img, model)
            if result == "unknown":
                st.error("Failed to classify the image.")
            else:
                st.success(f"The object in the image is classified as: {result}")

if __name__ == "__main__":
    run()
