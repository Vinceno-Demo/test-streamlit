import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from io import StringIO

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




# uploaded_file = st.file_uploader("Upload Image")
# image = Image.open(uploaded_file)
# st.image(image, caption='Input', use_column_width=True)
# img = Image.open(image)
# img.show()


uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    uploaded_file.read()
    image = Image.open(uploaded_file)
    if not image.mode == "RGB":
        image = image.convert('RGB')
    st.image(image, width=200)

    st.text(type(image))

latex_output = "\sum _ { i = 2 n + 3 m } ^ { 1 0 } i x"

st.latex(latex_output)
