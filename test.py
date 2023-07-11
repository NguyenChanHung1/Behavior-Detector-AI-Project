import streamlit as st
import numpy as np
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import time

class VideoCaptureTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model
        self.last_capture_time = 0

    def transform(self, frame):
        if time.time() - self.last_capture_time >= 5:
            # Capture image here
            image = frame.to_ndarray()
            cv2.imwrite('captured_image.jpg', image)
            self.last_capture_time = time.time()

        # Preprocess the frame if required
        preprocessed_frame = preprocess(frame)

        # Make predictions with your model
        predictions = self.model.predict(preprocessed_frame)

        # Add predictions to the frame
        frame_with_predictions = add_predictions(frame, predictions)

        return frame_with_predictions

# Function to preprocess the frame before feeding it into the model
def preprocess(frame):
    # Implement your frame preprocessing logic here
    # e.g., normalize pixel values, convert to appropriate format

    return frame

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
