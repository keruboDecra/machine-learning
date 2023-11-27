import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your saved model
model_path = 'models/tea_leaf_disease_model.h5'
model = load_model(model_path)

# Function to preprocess an image

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

# Function to make predictions using your model
def predict_tealeaf_disease(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions using your model
    # Replace the following line with your actual model inference code
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))

    return prediction

# Create a Gradio interface without the capture_session argument
iface = gr.Interface(
    fn=predict_tealeaf_disease,
    inputs=gr.Image(type="pil", label="Upload Your Image"),
    outputs="text",
    live=True  # Remove the capture_session argument
)


# Launch the Gradio interface
iface.launch()
