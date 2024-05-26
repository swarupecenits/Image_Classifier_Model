import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Image Classifier", page_icon=":camera_flash:")
st.markdown('<h1 class="title">Image Classification Model</h1>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="subheader">Upload an image of Vegetable or Fruit to identify</h2>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .title {
        color: white;
    }
    .subheader {
        color: white;
    }
    .footer {
        color: white;
    }
    .prediction {
        color: white;
    }
    .stApp {
        background: rgb(1,13,74);
        background: radial-gradient(circle, rgba(1,13,74,1) 0%, rgba(59,20,54,1) 100%);
        background-size: cover;
        background-repeat: no-repeat;
        backdrop-filter:blur(5px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model("Image_classifier.keras")

data_cat = [
    "Apple",
    "Banana",
    "Beetroot",
    "Bell pepper",
    "Cabbage",
    "Capsicum",
    "Carrot",
    "Cauliflower",
    "Chilli pepper",
    "Corn",
    "Cucumber",
    "Eggplant",
    "Garlic",
    "Ginger",
    "Grapes",
    "Jalepeno",
    "Kiwi",
    "Lemon",
    "Lettuce",
    "Mango",
    "Onion",
    "Orange",
    "Paprika",
    "Pear",
    "Peas",
    "Pineapple",
    "Pomegranate",
    "Potato",
    "Raddish",
    "Soy beans",
    "Spinach",
    "Sweetcorn",
    "Sweetpotato",
    "Tomato",
    "Turnip",
    "Watermelon",
]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = tf.keras.utils.load_img(uploaded_file, target_size=(180, 180))
    img_arr = tf.keras.utils.img_to_array(image)
    img_data = np.expand_dims(img_arr, axis=0)

    try:
        predict = model.predict(img_data)
        score = tf.nn.softmax(predict)

        st.image(uploaded_file, width=200)
        st.markdown(
            '<h2 class="subheader">Prediction Result</h2>', unsafe_allow_html=True
        )
        st.markdown(
            f'<p class="prediction">The Image is a {data_cat[np.argmax(score)]} with an accuracy of {np.max(score)*100:.2f}%</p>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown(
    '<p class="footer">Developed by Swarup Chanda 😎</p>', unsafe_allow_html=True
)
