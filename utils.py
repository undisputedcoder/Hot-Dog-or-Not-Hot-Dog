import tensorflow as tf

def load_and_prep_image(image):
    img = tf.io.decode_image(image, channels=3)
    img = tf.image.resize(img, size=[256,256])
    img = img/255.
    img = tf.expand_dims(img, axis=0)
    
    return img

def load_and_predict_model(image):
    model = tf.keras.models.load_model('models/model_2')
    prediction = model.predict(image)
    return prediction