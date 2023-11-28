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
  # Resize the image to the required size
    target_size = (224, 224)
    image = image.resize(target_size)
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Normalize pixel values (if required by your model)
    image_array = image_array / 255.0
    
    # Add any other preprocessing steps based on your model's requirements
    
    return image_array
    # return image

# Function to make predictions using your model
def predict_tealeaf_disease(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions using your model
    # Replace the following line with your actual model inference code
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    
    # Extract class names
    class_names = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']  # Replace with your actual class names

    # Find the predicted class index
    predicted_class_index = np.argmax(prediction[0])

    return class_names[predicted_class_index]

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_tealeaf_disease,
    inputs=gr.Image(type="pil", label="Upload Your Image"),
    outputs="text",
)

# Launch the Gradio interface
iface.launch()
