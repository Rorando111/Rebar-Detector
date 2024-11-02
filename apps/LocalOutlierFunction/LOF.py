import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import LocalOutlierFactor
import pickle
import streamlit as st

# Function to extract HOG features from an image
def extract_hog_features(image):
    image = cv2.resize(image, (128, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features

# Load the trained model
model_filename = 'apps/LocalOutlierFunction/lof_model.pkl'  # Update with your model path
with open(model_filename, 'rb') as model_file:
    lof_model = pickle.load(model_file)

# Streamlit application
st.title("Rebar Detection App")
st.write("Upload an image to check if it contains Rebar or Non-Rebar.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and process the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract HOG features from the image
    features = extract_hog_features(image)
    features = features.reshape(1, -1)  # Reshape for prediction

    # Predict using the LOF model
    prediction = lof_model.fit_predict(features)

    # Interpret the prediction
    if prediction[0] == 1:
        st.write("Prediction: **Rebar** (Normal)")
    else:
        st.write("Prediction: **Non-Rebar** (Anomaly)")

# Run the Streamlit app
if __name__ == "__main__":
    st.run()
