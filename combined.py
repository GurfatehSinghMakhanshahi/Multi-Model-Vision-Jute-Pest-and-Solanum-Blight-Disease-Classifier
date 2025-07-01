import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf


# Load your trained models
jute_pest_model = load_model('D:\\Major Project\\Image detection project\\mobilenetv2_best_model.keras')
plant_disease_model = load_model('D:\\Major Project\\Image detection project\\model (1).h5',
    compile=False)

# Define the image size, class names, and confidence threshold
IMG_WIDTH, IMG_HEIGHT = 224, 224
jute_class_names = [
    "Beet Armyworm", "Black Hairy", "Cutworm", "Field Cricket", "Jute Aphid",
    "Jute Hairy", "Jute Red Mite", "Jute Semilooper", "Jute Stem Girdler",
    "Jute Stem Weevil", "Leaf Beetle", "Mealybug", "Pod Borer", "Scopula Emissaria",
    "Termite", "Termite odontotermes (Rambur)", "Yellow Mite"
]
plant_disease_class_names = ['Potato Early blight', 'Potato Late blight', 'Potato healthy',
                             'Tomato Early blight', 'Tomato Late blight', 'Tomato healthy']

# Streamlit Multipage setup using sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a Model", ["Home", "Jute Pest Classifier", "Plant Disease Classifier"])


def predict_image(model, img_path, class_names):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if done during training

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])  # Get the index of the highest probability
    predicted_class_name = class_names[predicted_class_index]  # Map the index to the class name
    return predicted_class_name


# Home Page
if page == "Home":
    st.title("Project Information")
    st.write("""
        ### Welcome to the Jute Pest and Plant Disease Classifier
        This app leverages deep learning models to provide the following functionalities:

        1. **Jute Pest Classifier** - A model trained to identify 17 types of pests found on Jute crops. The model helps in early pest detection, potentially saving crops and reducing pesticide usage.

        2. **Plant Disease Classifier** - A model designed to diagnose diseases affecting crops like Potato and Tomato. It can classify healthy plants and predict diseases such as Early Blight and Late Blight, which are common in these crops.

       The accuracy and performance of these models can greatly assist in agriculture by automating the identification of crop pests and diseases.

        **Instructions:**
        - Use the sidebar to navigate to the specific classifier pages.
        - Upload an image on the respective pages and receive a prediction.

        ### Why Use This App?
        - **Pest Detection:** Early identification of pests helps farmers take timely action.
        - **Disease Diagnosis:** Prevent major crop losses by diagnosing diseases early.
        """)

# Jute Pest Classifier Page
elif page == "Jute Pest Classifier":
    st.title('Jute Pest Classifier 🦗🐛🐞')
    st.write('Upload an image to classify it.')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        # Save the uploaded file
        img_path = 'temp_jute_image.jpg'
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.image(img_path, caption='Uploaded Image', use_column_width=True)
        st.write("")

        # Predict and display the result
        predicted_class_name = predict_image(jute_pest_model, img_path, jute_class_names)
        st.write("Classification Class:", predicted_class_name)

# Plant Disease Classifier Page
elif page == "Plant Disease Classifier":
    st.title('Plant Disease Classifier 🌱🍅🥔')
    st.write('Upload an image to predict the plant disease.')

    img = st.file_uploader('Upload Image', type=['jpg'])
    if img is not None:
        st.image(img)

    if st.button('Classify'):
        img = Image.open(img)
        img_array = np.array(img)
        img_array = tf.image.resize(img_array, (256, 256))
        img_array = np.expand_dims(img_array, axis=0)
        pred = plant_disease_model.predict(img_array)
        pred = np.argmax(pred, axis=1)
        pred = plant_disease_class_names[pred[0]]
        st.info(pred)
