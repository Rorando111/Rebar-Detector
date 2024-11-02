import streamlit as st
import numpy as np
import cv2
import os
import pickle  # For loading the model
from skimage.feature import hog
from skimage import exposure
from sklearn.decomposition import PCA
from PIL import Image

# Load the model using pickle with error handling
model_path = 'apps/SVM/svm_model.pkl'  # Ensure this path is correct
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None

# Initialize PCA for reducing features to 512 components
pca = PCA(n_components=512)

# Function to extract HOG features from an image
def extract_hog_features(image):
    image = cv2.resize(image, (128, 64))  # Resize to HOG input size
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
    
    # Reshape features for PCA
    features = features.reshape(1, -1)
    
    # Apply PCA to reduce to 512 features
    features = pca.fit_transform(features)
    print("Extracted features shape after PCA:", features.shape)  # Debugging line
    return features

# Function to process and predict if an image is 'rebar' or 'non-rebar'
def processed_img(img_path, model):
    image = cv2.imread(img_path)
    if image is not None:
        features = extract_hog_features(image)  # Extract features
        try:
            prediction = model.predict(features)
            return "rebar" if prediction == 1 else "non-rebar"
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return "unknown"
    return "unknown"

# Main function for the Streamlit app
def run():
    st.title("Rebar Classification System")

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

        if st.button("Predict"):
            if model is not None:
                result = processed_img(save_image_path, model)
                if result == "unknown":
                    st.error("Failed to classify the image.")
                else:
                    st.success(f"The object in the image is classified as: {result}")
            else:
                st.error("Model is not loaded. Please check the model file.")

if __name__ == "__main__":
    run()
