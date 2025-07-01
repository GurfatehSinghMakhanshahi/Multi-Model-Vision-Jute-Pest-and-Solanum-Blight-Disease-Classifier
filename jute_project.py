import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model('D:\\Major Project\\Image detection project\\mobilenetv2_best_model.keras')

# Define the image size, class names, and confidence threshold
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Change this to the size used during training
class_names = [
    "Beet Armyworm", "Black Hairy", "Cutworm", "Field Cricket", "Jute Aphid",
    "Jute Hairy", "Jute Red Mite", "Jute Semilooper", "Jute Stem Girdler",
    "Jute Stem Weevil", "Leaf Beetle", "Mealybug", "Pod Borer", "Scopula Emissaria",
    "Termite", "Termite odontotermes (Rambur)", "Yellow Mite"
]
confidence_threshold = 0.5  # Set your desired threshold


# Define the function to make predictions
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization if you did that during training

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])  # Get the index of the highest probability
    confidence = predictions[0][predicted_class_index]  # Get the confidence of the predicted class

    if confidence < confidence_threshold:
        return "Image is not of Jute Pest. Please upload Correct Image"
    else:
        predicted_class_name = class_names[predicted_class_index]  # Map the index to the class name
        return predicted_class_name


# Streamlit UI
st.title('Jute Pest Classifier 🦗🐛🐞')
st.write('Upload an image to classify it.')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file
    img_path = 'temp_image.jpg'
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.image(img_path, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Predict and display the result
    predicted_class_name = predict_image(img_path)
    st.write("Classification Class:", predicted_class_name)
