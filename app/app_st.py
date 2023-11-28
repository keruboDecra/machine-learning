import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# python -m pip install {package_name}.

# Load the saved model
model = load_model('models/tea_leaf_disease_model.h5')

# Define class names
class_names = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']

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
    st.title("Tea Leaf Disease Classification")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")


    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            predictions = predict_image(image)
            st.write("Predictions:")
            st.write(predictions)

if __name__ == '__main__':
    main()