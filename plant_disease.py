import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

res = ['Potato Early blight', 'Potato Late blight', 'Potato healthy','Tomato Early blight', 'Tomato Late blight',  'Tomato healthy']
# Load the model without compiling (useful if only for inference)
model = tf.keras.models.load_model(
    'D:\\Major Project\\Image detection project\\model (1).h5',
    # 'C:\\Jute Project\\model (1).h5',
    compile=False
)

st.title('Plant Disease Prediction')

img = st.file_uploader('Upload Image', type=['jpg'])
if img is not None:
    st.image(img)
if st.button('Classify'):
    img = Image.open(img)
    img_array = np.array(img)
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred = np.argmax(pred, axis=1)
    pred = res[pred[0]]
    st.info(pred)
