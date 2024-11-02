import streamlit as st
import numpy as np
import cv2
import os
import pickle
from skimage.feature import hog
from skimage import exposure
from PIL import Image

# Load the KNN model using pickle
model_path = '/mount/src/rebar-detector/apps/knn_model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Function to extract HOG features from an image
def extract_hog_features(image):
    image = cv2.resize(image, (128, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return features

# Function to process and predict with a threshold
def processed_img_with_threshold(img_path, model, threshold=0.5):
    image = cv2.imread(img_path)
    if image is not None:
        features = extract_hog_features(image).reshape(1, -1)
        distances, _ = model.kneighbors(features)
        avg_distance = np.mean(distances)

        if avg_distance < threshold:
            prediction = model.predict(features)
            return "rebar" if prediction[0] == 1 else "non-rebar"
        else:
            return "non-rebar (outlier)"
    return "unknown"

# Main function for the Streamlit app
def run():
    st.title("Rebar Classification System with Threshold")

    img_file = st.file_uploader("Upload an Image for Classification", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, use_column_width=True)

        # Save uploaded image to a temporary directory
        upload_dir = './uploaded_images/'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        save_image_path = os.path.join(upload_dir, img_file.name)
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        threshold = st.slider("Set classification threshold", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

        if st.button("Predict"):
            result = processed_img_with_threshold(save_image_path, model, threshold)
            if result == "unknown":
                st.error("Failed to classify the image.")
            else:
                st.success(f"The object in the image is classified as: {result}")

if __name__ == "__main__":
    run()
