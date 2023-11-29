import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# from captum.attr import IntegratedGradients

# python -m pip install {package_name}.

# Load the saved model
model = load_model('tea_leaf_disease_model.h5')

# Define class names
class_names = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']

@st.cache 
# Define image preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    # Add more preprocessing steps as needed
    return img

def predict_image(img):
    # Preprocess the image
    img_preprocessed = preprocess_image(img)
    img_4d = img_preprocessed.reshape(1, 224, 224, 3)  # Add an extra dimension for batch size
    
    # Ensure the model output is a probability distribution using softmax
    predictions = model.predict(img_4d)[0]
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

    # Map probabilities to class names
    predictions_dict = {class_names[i]: round(float(probabilities[i])) for i in range(len(class_names))}
    
    return predictions_dict

def main():
    st.title("Tea Leaf Disease Classification :tea: :coffee:")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add a condition to check if the uploaded image is relevant to the task
        if is_relevant_image(image):
            if st.button("Classify"):
                predictions = predict_image(image)
                st.write("Predictions:")
                st.write(predictions)
        else:
            st.warning("The uploaded image is not relevant to the task. Please choose an image of tea leaves.")

def is_relevant_image(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a green color range in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a mask to extract green regions
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Calculate the percentage of green pixels in the image
    green_percentage = np.sum(green_mask) / (image.shape[0] * image.shape[1])

    # Adjust the threshold based on your specific requirements
    threshold = 0.1

    # Return True if the green percentage is above the threshold, indicating a relevant image
    return green_percentage > threshold

if __name__ == '__main__':
    main()
