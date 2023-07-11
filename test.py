import streamlit as st
import numpy as np
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import time

import keras
import keras.preprocessing
from keras.models import load_model
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import pickle
from tensorflow.keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# Function to preprocess the frame before feeding it into the model
def preprocess(img_path):
  img = image.load_img(img_path, target_size=(48, 48))
  img = image.img_to_array(img)
  img = np.reshape(img, (1, 48, 48, 3))
  img = preprocess_input(img)
  return img
class VideoCaptureTransformer(VideoTransformerBase):
    
    # img = load_image_for_model(img_test_path)
    # def __init__(self, model):
    #     self.model = model
    #     self.last_capture_time = 0

    def transform(self, frame):

        if time.time() - self.last_capture_time >= 5:
            # Capture image here
            image = frame.to_ndarray()
            cv2.imwrite('captured_image.jpg', image)
            self.last_capture_time = time.time()

        # Preprocess the frame if required
        emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
        img = preprocess("Behavior-Detector-AI-Project/captured_image.jpg")

        # Make predictions with your model
        loaded_model = load_model("C:/Users/Hi/Downloads/face_expression_resnet50_2.h5")
        lbl_predictions = loaded_model.predict(img, verbose=False)
        for lbl in lbl_predictions:
            max_idx = np.argmax(lbl)
            lbl_pred = max_idx
        #print(emotion_dict[lbl_pred])

        # Add predictions to the frame
        # frame_with_predictions = add_predictions(frame, predictions)

        return emotion_dict[lbl_pred]



# Function to add predictions to the frame
def add_predictions(frame, predictions):
    # Implement your logic to add predictions to the frame
    # e.g., draw bounding boxes, labels, etc.

    return frame

# Modify your Streamlit app to use the VideoCaptureTransformer
def app():
    # Load your model
    model = load_model()

    # Set the title of the app
    st.title("Webcam")

    try:
        # Use the webrtc_streamer function to capture video from the default webcam
        ctx = webrtc_streamer(
            key="example",
            video_transformer_factory=VideoCaptureTransformer,
            async_transform=True,
            transformer_params=(model,),
        )

        # Add a button to toggle table visibility
        if ctx.video_transformer:
            if "show_table" not in st.session_state:
                st.session_state.show_table = False
            if not st.session_state.show_table:
                if st.sidebar.button("Show Table"):
                    st.session_state.show_table = True
            
            # Display table if button is clicked
            if st.session_state.show_table:
                st.write("Table")
                st.table({
                    'Full Name': ['Text 1', 'Text 2'],
                    'Emotion': ['Text 3', 'Text 4']
                })

        # Handle webcam screen display
        if ctx.video_transformer:
            frame_with_predictions = ctx.video_transformer.frame_out
            st.image(frame_with_predictions, channels="BGR")

    except Exception as e:
        print(e)

# Function to load your model
def load_model():
    # Implement your logic to load the model
    # e.g., keras.models.load_model()

    return model

if __name__ == "__main__":
    app()
