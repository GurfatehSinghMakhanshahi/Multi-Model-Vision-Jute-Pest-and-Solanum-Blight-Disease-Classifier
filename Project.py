import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model('C:\\Users\\luxma\\Desktop\\Image detection project\\mobilenetv2_best_model.keras')


# Define the function to make predictions
IMG_WIDTH, IMG_HEIGHT = 224, 224
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization if you did that during training

    predictions = model.predict(img_array)
    return predictions


# Streamlit UI
st.title('Jute Pest Classifier')
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
    predictions = predict_image(img_path)
    st.write("Prediction:", predictions)
