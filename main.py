import streamlit as st
from utils import *

def make_prediction(image, class_names):
    img = load_and_prep_image(image=image)
    prediction = load_and_predict_model(img)
    pred_class = class_names[int(tf.round(prediction)[0][0])]

    return pred_class

st.title("Hot Dog or Pizza 🌭 🍕")
st.subheader("Detect if an image is either a hot dog or pizza")

uploaded_file = st.file_uploader("Upload an image of a pizza or hot dog", type=["png", "jpeg", "jpg"])

if not uploaded_file:
    st.warning("No file has been uploaded.")
    st.stop()
else:
    uploaded_image = uploaded_file.read()
    st.image(uploaded_image, width=350)
    pred_btn = st.button("Predict")

if pred_btn:
    pred_btn = True

if pred_btn:
    class_names = ["Hot Dog", "Pizza"]
    prediction = make_prediction(uploaded_image, class_names)
    st.markdown(f"The model predicted **{prediction}**")