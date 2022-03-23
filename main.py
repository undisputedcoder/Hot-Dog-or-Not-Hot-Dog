import streamlit as st
from utils import *

def make_prediction(image, class_names):
    img = tf.io.decode_image(image, channels=3)
    img = tf.image.resize(img, size=[256,256])
    img = img/255.
    img = tf.expand_dims(img, axis=0)

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
    st.image(uploaded_image)
    pred_btn = st.button("Predict")

if pred_btn:
    pred_btn = True

if pred_btn:
    class_names = ["Hot Dog", "Pizza"]
    prediction = make_prediction(uploaded_image, class_names)
    st.markdown(f"The model predicted the image is a **{prediction}**")
    print(f"Prediction: {prediction}")

    feedback = st.radio("Did the model predict correctly?", ('👍 Yes', '👎 No'))

    if feedback == 'Yes':
        st.write("Thanks for the feedback")
    elif feedback == 'No':
        st.write("Darn. I guess the model needs improving")
