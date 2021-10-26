import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


st.markdown("<h1 style='text-align: center;'>Squirrel VS Tortoise</h1>", unsafe_allow_html=True)

model = tf.keras.models.load_model("squirrelvstortoise[72]data.h5")
### load file
file = st.file_uploader("Choose a image file of squirrel or tortoise", type="jpg")

classes = {'squirrel': 0, 'tortoise': 1}

if file is not None:
    image = Image.open(file)

    st.image(
        image,
    )

    img_array = np.array(image)
    img = tf.image.resize(img_array, size=(224,224))
    img = img/255
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    st.title("Predicted Label for the image is {}".format(list(classes.keys())[list(classes.values()).index(int(tf.round(prediction)))]))