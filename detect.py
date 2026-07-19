from PIL import Image, ImageOps
import torch
import cv2
import numpy as np
import os
import sys
from torchvision.models import inception_v3, Inception_V3_Weights
from functions import import_and_predict, build_model

label = ""
frame = None

NUM_CLASSES = 3
device = torch.device("cpu")
weights = Inception_V3_Weights.DEFAULT

pre_trained_model = inception_v3(weights=weights, aux_logits=True)
for param in pre_trained_model.parameters():
    param.requires_grad = False
model = build_model(pre_trained_model, NUM_CLASSES)
model.to(device)

model.load_state_dict(torch.load("./models/best_model.pth"))

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("Camera OK")
else:
    cap.open()


while True:
    ret, original = cap.read()

    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')

    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    prediction = import_and_predict(image, model)
    # print(prediction)

    # change rps to apple, banana, orange and for fruit nutritional checker.
    # Detects fruits and decides if fruit is useful in user's current condition (Anaemia, Bulking, etc.)
    if np.argmax(prediction) == 0:
        predict = "It is a paper!"
    elif np.argmax(prediction) == 1:
        predict = "It is a rock!"
    else:
        predict = "It is a scissor!"

    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
