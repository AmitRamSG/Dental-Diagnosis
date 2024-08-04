import streamlit as slt
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

MODEL=keras.models.load_model(r"C:\Users\kedar\Downloads\VGG16.h5")
upload_image = slt.file_uploader(label='Upload image', type=['png', 'jpg','jpeg'],accept_multiple_files=False)

if upload_image is not None:

    image=Image.open(upload_image)

    converted_img = np.array(image.convert('RGB'))

    img = cv2.resize(converted_img, dsize=(256,256))

    img_reshape = np.reshape(img,[1,256,256,3])

    y_predict = np.argmax(MODEL.predict(img_reshape), axis=1)

    slt.text(y_predict)