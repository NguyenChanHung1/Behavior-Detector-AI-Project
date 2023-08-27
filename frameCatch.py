import streamlit as st
import numpy as np
import cv2
import keras
import keras.preprocessing
from keras.models import load_model
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import pickle
from tensorflow.keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

def get_expression(img_test_path):
    emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
    model = load_model("C:/Users/Hi/Downloads/face_expression_resnet50_2.h5")
    # img = keras.preprocessing.image.load_img(img_test_path, target_size=(224, 224))
    img = load_image_for_model(img_test_path)
    

    lbl_predictions = model.predict(img, verbose=False)
    for lbl in lbl_predictions:
        max_idx = np.argmax(lbl)
        lbl_pred = max_idx
    print(emotion_dict[lbl_pred])
    # print(lbl_predictions)

def load_image_for_model(img_path):
  img = image.load_img(img_path, target_size=(48, 48))
  img = image.img_to_array(img)
  img = np.reshape(img, (1, 48, 48, 3))
  img = preprocess_input(img)
  return img
    

if __name__ == "__main__":
    get_expression("Behavior-Detector-AI-Project/face_picture.jpg")