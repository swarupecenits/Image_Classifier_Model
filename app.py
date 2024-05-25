import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title='Image Classifier', page_icon=':camera_flash:')
st.title('Image Classification Model')
st.subheader('Upload an image to identify')

model = load_model('Image_classifier.keras')

data_cat = ['Apple', 'Banana', 'Beetroot', 'Bell pepper', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower',
            'Chilli pepper', 'Corn', 'Cucumber', 'Eggplant', 'Garlic', 'Ginger', 'Grapes', 'Jalepeno',
            'Kiwi', 'Lemon', 'Lettuce', 'Mango', 'Onion', 'Orange', 'Paprika', 'Pear', 'Peas', 'Pineapple',
            'Pomegranate', 'Potato', 'Raddish', 'Soy beans', 'Spinach', 'Sweetcorn', 'Sweetpotato', 'Tomato',
            'Turnip', 'Watermelon']

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = tf.keras.utils.load_img(uploaded_file, target_size=(180, 180))
    img_arr = tf.keras.utils.img_to_array(image)
    img_data = np.expand_dims(img_arr, axis=0)

    predict = model.predict(img_data)
    score = tf.nn.softmax(predict)

    st.image(uploaded_file, width=200)
    st.subheader('Prediction Result')
    st.write(f'The Image is a {data_cat[np.argmax(score)]} with an accuracy of {np.max(score)*100:.2f}%')


