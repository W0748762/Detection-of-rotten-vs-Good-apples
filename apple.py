import tensorflow as tf
import numpy as np
import streamlit as st
import requests
from io import BytesIO
from keras.preprocessing import image
import os
import keras

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Apple Identifier")
st.text("upload Apple image")

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('apple')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['Good Apples', 'Rotten Apples']

def scale(image):
  image = tf.cast(image, tf.float32)
  image /= 255.0

  return tf.image.resize(image,[148,148])

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = scale(img)
  return np.expand_dims(img, axis=0)


uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg"])

if uploaded_file is not None:
    data = uploaded_file.read()
    st.image(data, caption='Uploaded Image.', use_column_width=True)

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      im = decode_img(data)
      proba = model.predict(im.reshape(1,148,148,3))
      top = np.argsort(proba[0])
      st.write("This image most likely belongs to {} with a {:.2f} percent confidence.".format(classes[np.argmax(top)], 100 * np.max(top)))      
    st.write("")
