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
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import asyncio

loaded_model = load_model("C:/Users/Hi/Downloads/face_expression_resnet50_2.h5")


def preprocess(img_path):
        img = image.load_img(img_path, target_size=(48, 48))
        img = image.img_to_array(img)
        img = np.reshape(img, (1, 48, 48, 3))
        img = preprocess_input(img)
        return img
class VideoCaptureTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_capture_time = 0

    def transform(self, frame):
        global emotion_dict
        global emotion
        global current_emotion

        if time.time() - self.last_capture_time >= 2:
            # Capture image here
            image = frame.to_ndarray()
            cv2.imwrite("C:/AI project/app/Behavior-Detector-AI-Project/captured_image.jpg", image)
            self.last_capture_time = time.time()

        # Preprocess the frame if required
        emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
        img = preprocess("C:/AI project/app/Behavior-Detector-AI-Project/captured_image.jpg")

        # Load the model
        global loaded_model

        # Make predictions with the model
        lbl_predictions = loaded_model.predict(img, verbose=False)
        lbl_pred = np.argmax(lbl_predictions)
        emotion = emotion_dict[lbl_pred]
        current_emotion = emotion
        
        # Add predictions to the frame
        frame_with_predictions = frame.to_ndarray()
        frame_with_predictions = cv2.cvtColor(frame_with_predictions, cv2.COLOR_BGR2RGB)
        print(emotion)
        cv2.putText(frame_with_predictions, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Perform zoom-out transformation on the frame
        zoom_out_factor = 2
        width = int(frame_with_predictions.shape[1] / zoom_out_factor)
        height = int(frame_with_predictions.shape[0] / zoom_out_factor)
        frame_zoomed = cv2.resize(frame_with_predictions, (width, height))

        return frame_zoomed

# Modify your Streamlit app to use the VideoCaptureTransformer
def app():
    # Set the title of the app
    st.title("Webcam")
    

    try:
        # Use the webrtc_streamer function to capture video from the default webcam
        webrtc_ctx = webrtc_streamer(
            key="example",
            video_transformer_factory=VideoCaptureTransformer,
            async_transform=True,
        )

        # Handle webcam screen display
        if webrtc_ctx.video_transformer:
            frame_with_predictions = webrtc_ctx.video_transformer.frame_out
            st.image(frame_with_predictions, channels="RGB")
            emotion = webrtc_ctx.video_transformer.current_emotion

            # Set the initial value of current_emotion if it is not set yet
            if 'current_emotion' not in st.session_state:
                st.session_state.current_emotion = emotion

            # Update the current emotion if it has changed
            if emotion != st.session_state.current_emotion:
                st.session_state.current_emotion = emotion

            st.session_state.show_table = True
            # Display table if button is clicked
            if st.session_state.show_table:
                st.write("Table")
                st.table({
                    'Emotion': [st.session_state.current_emotion]
                })
            # frame_with_predictions = webrtc_ctx.video_transformer.frame_out
            # st.image(frame_with_predictions, channels="BGR")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    app()
