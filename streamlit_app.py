# import streamlit as st
# from PIL import Image
# import numpy as np
# import pandas as pd
# from io import StringIO


# import torch
# import logging
# import argparse
# from PIL import Image
# from transformers import AutoProcessor
# from transformers import VisionEncoderDecoderModel


# img_file_buffer = st.camera_input("Take a picture")

# if img_file_buffer is not None:
#     # To read image file buffer as a PIL Image:
#     img = Image.open(img_file_buffer)
#     st.write('Show image:')
#     st.image(img, width=200)




# uploaded_file = st.file_uploader("Choose a image file", type="jpg")
# if uploaded_file is not None:
#     # Convert the file to an opencv image.
#     uploaded_file.read()
#     image = Image.open(uploaded_file)
#     if not image.mode == "RGB":
#         image = image.convert('RGB')
#     st.image(image, width=200)



# latex_output = "\sum _ { i = 2 n + 3 m } ^ { 1 0 } i x"
# st.write('Latex code:', latex_output)
# st.latex(latex_output)



----------------------------------------------------


import torch
import logging
import argparse
from PIL import Image
from transformers import AutoProcessor
from transformers import VisionEncoderDecoderModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Get the device
if torch.cuda.is_available():    
    device = torch.device("cuda")
    logger.info("There are {} GPU(s) available.".format(torch.cuda.device_count()))
    logger.info('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))
else:
    logger.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Init model
model = VisionEncoderDecoderModel.from_pretrained(
    facebook/nougat-base
).to(device)

# Init processor
processor = AutoProcessor.from_pretrained(facebook/nougat-base)

uploaded_file = st.file_uploader("Choose a image file", type="jpg")
if uploaded_file is not None:
    # Convert the file to an opencv image.
    uploaded_file.read()
    image = Image.open(uploaded_file)
    if not image.mode == "RGB":
        image = image.convert('RGB')
    st.image(image, width=200)
    
# # Load image
# image = Image.open(args.input_image)
# if not image.mode == "RGB":
#     image = image.convert('RGB')

pixel_values = processor.image_processor(
    image,
    return_tensors="pt",
    data_format="channels_first",
).pixel_values
task_prompt = processor.tokenizer.bos_token
decoder_input_ids = processor.tokenizer(
    task_prompt,
    add_special_tokens=False,
    return_tensors="pt"
).input_ids

# Generate LaTeX expression
with torch.no_grad():
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_length,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
sequence = processor.tokenizer.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(
        processor.tokenizer.eos_token, ""
    ).replace(
        processor.tokenizer.pad_token, ""
    ).replace(processor.tokenizer.bos_token,"")

st.write('Latex code:', sequence)
st.latex(sequence)

