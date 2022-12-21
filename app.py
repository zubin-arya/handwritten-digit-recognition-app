import streamlit as st
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1" #force predict to run on cpu
model = keras.models.load_model('savedModel')
stroke_width = st.sidebar.slider(" Stroke Width: ", 1, 30, 25)
stroke_color = st.sidebar.color_picker(" Stroke color hex :")
background_color = st.sidebar.color_picker("Background color hex: ","#eee")
#background_image = st.sidebar.file_uploader("Background image:", type = ["png", "jpg"])
drawing_mode = st.sidebar.selectbox("Drawing Tool: ",("freedraw", "line", "rect", "circle", "transform", "polygon"))
realtime_update = st.sidebar.checkbox("Update in realtime", True)

#Create a canvas component
canvas_result = st_canvas(
    fill_color = "rgba(255, 165, 0, 0.3)",
    stroke_width = stroke_width,
    stroke_color = stroke_color,
    background_color = background_color,
    # background_image = Image.open(background_image) if background_image else None,
    update_streamlit = realtime_update,
    height = 200,
    width = 200,
    drawing_mode = drawing_mode,
    display_toolbar = st.sidebar.checkbox("Display toolbar", True),
    key = "full_app",
    )

#Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    image = canvas_result.image_data
    image1 = image.copy()
    image1 = image1.astype('uint8')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (28, 28))
    st.image(image1)
    image1.resize(1, 28, 28, 1)
    st.title(np.argmax(model.predict(image1)))
    
if canvas_result.json_data is not None:
    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
    

