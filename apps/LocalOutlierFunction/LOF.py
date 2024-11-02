import streamlit as st
import numpy as np
import cv2
import pickle
from skimage.feature import hog
from PIL import Image

# Load the LOF model using pickle
model_path = 'apps/LocalOutlierFunction/lof_model.pkl'  # Update this path as necessary
with open(model_path, 'rb') as model_file:
    lof_model = pickle.load(model_file)

# Function to extract HOG features from an image
def extract_hog_features(image):
    image = cv2.resize(image, (128, 64))  # Ensure the same dimensions used during training
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=True)
    return features

# Function to process and predict
def process_and_predict(img, model):
    image = np.array(img)
    if image is not None:
        features = extract_hog_features(image).reshape(1, -1)  # Extract HOG features
        prediction = model.fit_predict(features)  # Use the LOF model to predict

        # Convert prediction to readable format
        if prediction[0] == 1:
            return "rebar"
        elif prediction[0] == -1:
            return "non-rebar (outlier)"
        else:
            return "unknown"
    return "unknown"

# Main function for the Streamlit app
def run():
    st.title("Rebar Classification System")

    img_file = st.file_uploader("Upload an Image for Classification", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        if st.button("Predict"):
            result = process_and_predict(img, lof_model)
            if result == "unknown":
                st.error("Failed to classify the image.")
            else:
                st.success(f"The object in the image is classified as: {result}")

if __name__ == "__main__":
    run()
