import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision.models import inception_v3, Inception_V3_Weights
from functions import build_model


def import_and_predict(image_data, model):
        size = (299, 299)
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)     #prepare image with antialiasing
        image = image.convert('RGB')    #convert image to RGB, Red Green Blue format
        image = np.asarray(image)       #convert image into array
        image = (image.astype(np.float32) / 255.0)      #create image array matrix
        image = np.transpose(image, (2, 0, 1))          #HWC -> CHW
        img_reshape = torch.from_numpy(image).unsqueeze(0)   #add batch dimension

        model.eval()
        with torch.no_grad():
            prediction = model(img_reshape)
            prediction = prediction.numpy()

        return prediction


NUM_CLASSES = 3
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
weights = Inception_V3_Weights.DEFAULT

pre_trained_model = inception_v3(weights=weights, aux_logits=True)
for param in pre_trained_model.parameters():
    param.requires_grad = False
model = build_model(pre_trained_model, NUM_CLASSES)
model.to(device)

model.load_state_dict(torch.load("./models/best_model.pth")) #loading a trained model

# change rps to apple, banana, orange and for fruit nutritional checker.
# Detects fruits and decides if fruit is useful in user's current condition (Anaemia, Bulking, etc.)

st.write("""
         # Rock-Paper-Scissor Hand Sign Prediction 
         """
         )

st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("It is a paper!")
    elif np.argmax(prediction) == 1:
        st.write("It is a rock!")
    else:
        st.write("It is a scissor!")

    st.text("Probability (0: Paper, 1: Rock, 2: Scissor)")
    st.write(prediction)