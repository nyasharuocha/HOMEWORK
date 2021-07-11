import glob
import os
import shutil
import glob
import cv2
import numpy as np
import streamlit as st
import tempfile
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from matplotlib.pyplot import gray
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

st.title("VIDEO MODEL USING VGG16")
if not os.path.exists('frames'):
    os.makedirs('frames')
else:
    shutil.rmtree('frames')
    os.makedirs('frames')

f = st.file_uploader("Upload a Video", type=['mp4'])
count = 0

if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())

    vf = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN

    while True:
        ret, frame = vf.read()

        if not ret:
            break
        name = './frames/*' + str(count) + '.jpg'
        cv2.imwrite(name, frame)
        img = './frames/*' + str(count) + '.jpg'
        img = image.load_img(img, color_mode='rgb', target_size=(224, 224))
        imageArray = image.img_to_array(img)
        imageArray = np.expand_dims(imageArray, axis=0)
        imageArray = preprocess_input(imageArray)
        features = model.predict(imageArray)
        img = cv2.putText(frame, str(decode_predictions(features)[0][0][1]), (30, 55), font, fontScale=font_scale,
                          color=(0, 255, 0), thickness=4)
        # img = cv2.imwrite('./frames/*' + str(count) + '.jpg', frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img)
        count += 1

name = st.text_input(label="Select Frame")

path = './frames/*'

for name in glob.glob('./frames/*' + str(count) + '.jpg'):
    st.frame.image(path + str(name))


