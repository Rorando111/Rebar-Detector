import streamlit as st
import numpy as np
import cv2
import pickle
from skimage.feature import hog
from skimage import exposure
from PIL import Image

# Function to extract HOG features from an image
def extract_hog_features(image):
    image = cv2.resize(image, (128, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features

# Load the Isolation Forest model using pickle
model_path = '/content/drive/MyDrive/Datasets/models/isolation_forest_model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict using the Isolation Forest model
def predict_image_isof(image, model):
    image = np.array(image)
    features = extract_hog_features(image).reshape(1, -1)
    pred = model.predict(features)
    return "rebar" if pred[0] == 1 else "non-rebar"

# Main function for the Streamlit app
def run():
    st.title("Rebar Classification with Isolation Forest")

    img_file = st.file_uploader("Upload an Image for Classification", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, use_column_width=True)

        if st.button("Predict"):
            result = predict_image_isof(img, model)
            st.success(f"The object in the image is classified as: {result}")

if __name__ == "__main__":
    run()
