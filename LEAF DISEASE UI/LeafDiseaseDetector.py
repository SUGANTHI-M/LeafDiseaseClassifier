import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import gradio as gr

mangoclasses = [ 'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
grapeclasses = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

def mangoleafdetector(input_image):
    img = input_image.resize((224, 224))
    img = img.convert("RGB")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    model = load_model("mangoNet.h5")
    prediction = model.predict(img)
    return f'{mangoclasses[np.argmax(prediction)]}'

def grapeleafdetector(input_image):
    img = input_image.resize((160,160))
    img = img.convert("RGB")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    model = load_model("grapeNet.h5")
    prediction = model.predict(img)
    return f'{grapeclasses[np.argmax(prediction)]}'


iface1 = iface = gr.Interface(fn=mangoleafdetector, 
                     inputs=gr.inputs.Image(type="pil"),
                     outputs=gr.outputs.Textbox(),
                     title='LEAF DISEASE CLASSIFIER',allow_flagging="never")

iface2 = iface = gr.Interface(fn=grapeleafdetector, 
                     inputs=gr.inputs.Image(type="pil"),
                     outputs=gr.outputs.Textbox(),
                     title='LEAF DISEASE CLASSIFIER',allow_flagging="never")

demo = gr.TabbedInterface([iface1, iface2], ["Mango", "Grape"])
demo.launch(share=True)