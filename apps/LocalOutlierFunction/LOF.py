import streamlit as st
import numpy as np
import cv2
import os
import pickle  # Keep using pickle for loading the model
from skimage.feature import hog
from skimage import exposure
from sklearn.neighbors import LocalOutlierFactor
from PIL import Image

# Load the model using pickle
model_path = 'apps/LocalOutlierFunction/lof_model.pkl'  # Ensure this path is correct
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

# Function to process and predict if an image is 'rebar' or 'non-rebar'
def processed_img(image, model):
    features = extract_hog_features(image).reshape(1, -1)
    
    # Use the model to predict
    prediction = model.predict(features)  # Use the fitted model to predict
    return "rebar" if prediction[0] == 1 else "non-rebar (outlier)"

# Main function for the Streamlit app
def run():
    st.title("Rebar Classification System Using Local Outlier Factor")

    img_file = st.file_uploader("Upload an Image for Classification", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, use_column_width=True)

        if st.button("Predict"):
            result = processed_img(np.array(img), model)  # Pass the image directly
            st.success(f"The object in the image is classified as: {result}")

if __name__ == "__main__":
    run()
