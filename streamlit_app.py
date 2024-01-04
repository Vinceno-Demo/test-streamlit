import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from io import StringIO
import cv2

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Check the type of img_array:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(img_array))

    # Check the shape of img_array:
    # Should output shape: (height, width, channels)
    st.write(img_array.shape)




uploaded_file = st.file_uploader("Upload Image")
image = Image.open(uploaded_file)
st.image(image, caption='Input', use_column_width=True)
img_array = np.array(image)
cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
