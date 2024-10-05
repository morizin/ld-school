# %%writefile tb-school/app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image
from tensorflow.keras.optimizers import Adam
import numpy as np

# Suppress warnings if you'd like
# import warnings
# warnings.filterwarnings('ignore')

def create_model():
    conv_base = EfficientNetB0(include_top = False, weights = None,
                               input_shape = (512, 512, 3))
    model = conv_base.output
    model = layers.GlobalAveragePooling2D()(model)
    model = layers.Dense(5, activation = "softmax")(model)
    model = models.Model(conv_base.input, model)

    model.compile(optimizer = Adam(learning_rate = 0.001),
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["acc"])
    return model

model = create_model()
model.load_weights('EfNetB0_275_16.h5')

# # Load your pre-trained model
# model = tf.keras.models.load_model('Cassava Leaf Model 1.h5')

# Function to preprocess the image and make predictions
def lung_defect(img):

    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Preprocessing the image to fit the model input shape
    img = tf.image.resize(img, [512, 512])
    img = img[None, ...]
    # img_array = img_array / 255.0  # Assuming the model expects the input in this range
    # img_array = img_array.reshape((1, 224, 224, 3))  # Adjusting to the input shape

    # Make a prediction
    prediction = model.predict(img).tolist()[0]
    class_names = ['Bacterial Blight', 'Brown Streak Disease', 'Green Mottle', 'Mosaic Disease', 'Healthy']

    # Returning a dictionary of class names and corresponding predictions
    return {class_names[i]: float(prediction[i]) for i in range(5)}

# Streamlit user interface
st.title('Leaf Disease Classification')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    predictions = lung_defect(image)

    # Display the predictions as a bar chart
    st.bar_chart(predictions)
