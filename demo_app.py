import streamlit as st
from PIL import Image

import os
import imghdr
from io import BytesIO
import base64

# style image paths:
root_style = "./images/style-images"

import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16
import streamlit as st

# we will use the conecpt of caching here that is once a user has used a particular model instead of loading
# it again and again everytime they use it we will cache the model.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading a model


@st.cache
def load_model(model_path):

    with torch.no_grad():
        style_model = TransformerNet()  # transformer_net.py contain the style model
        state_dict = torch.load(model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        return style_model

# we need the content image and the the style model that we have loaded
# with load_model function


@st.cache
def stylize(style_model, content_image, output_image):

    # if the content image is a path then
    if type(content_image) == "str":
        content_image = utils.load_image(
            content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)

    # to treat a single image like a batch
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = style_model(content_image).cpu()

    # output image here is the path to the output image
    img = utils.save_image(output_image, output[0])
    return img


# download image function
def get_image_download_link(img, file_name, style_name):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a style = "color:black" href="data:file/jpg;base64,{img_str}" download="{style_name+"_"+file_name+".jpg"}"><input type="button" value="Download"></a>'
    return href


st.markdown("<h1 style='text-align: center; color: Blue;'>Neural Style Transfer</h1>",
            unsafe_allow_html=True)



main_bg = "./images/pyto.png"
main_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# creating a side bar for picking the style of image
style_name = st.sidebar.selectbox(
    'Select Style',
    ("candy", "mosaic", "rain_princess",
     "udnie", "tg", "demon_slayer", "ben_giles", "ben_giles_2")
)
path_style = os.path.join(root_style, style_name+".jpg")


# Upload image functionality
img = None
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

show_file = st.empty()
col1, col2 = st.columns(2)
# checking if user has uploaded any file
if not uploaded_file:
    show_file.info("Please Upload an Image")
else:
    img = Image.open(uploaded_file)
    # check required here if file is an image file
    col1.image(img, caption='Uploaded Image.', use_column_width=True)
    col2.image(path_style, caption='Style Image', use_column_width=True)


extensions = [".png", ".jpeg", ".jpg"]

if uploaded_file is not None and any(extension in uploaded_file.name for extension in extensions):

    name_file = uploaded_file.name.split(".")
    root_model = "./saved_models"
    model_path = os.path.join(root_model, style_name+".pth")

    img = img.convert('RGB')
    input_image = img

    root_output = "./images/output-images"
    output_image = os.path.join(
        root_output, style_name+"_op.jpg")

    stylize_button = st.button("Stylize")

    if stylize_button:
        model = load_model(model_path)
        stylized = stylize(model, input_image, output_image)
        # displaying the output image
        st.write("### Output Image")
        # image = Image.open(output_image)
        st.image(stylized, width=400, use_column_width=True)
        st.markdown(get_image_download_link(
            stylized, name_file[0], style_name), unsafe_allow_html=True)
