import os
import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog
from sklearn.neighbors import LocalOutlierFactor
import pickle

# Function to extract HOG features from an image
def extract_hog_features(image):
    image = cv2.resize(image, (128, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features

# Load the trained LOF model
model_filename = 'apps/LocalOutlierFunction/lof_model.pkl'  # Update this path accordingly
with open(model_filename, 'rb') as model_file:
    lof_model = pickle.load(model_file)

# Streamlit app
st.title("Rebar vs Non-Rebar Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract features
    features = extract_hog_features(image).reshape(1, -1)  # Reshape for a single sample

    # Make prediction
    prediction = lof_model.fit_predict(features)

    # Convert prediction to readable format
    if prediction[0] == 1:  # Corrected to access the prediction array element
        st.write("Prediction: Rebar")
    else:
        st.write("Prediction: Non-Rebar")
