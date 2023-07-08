import streamlit as st
import av
import streamlit.logger
import numpy as np
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Define a video transformer that zooms out the frame
class ZoomOutTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Perform zoom-out transformation on the frame
        zoom_out_factor = 2
        width = int(frame.shape[1] / zoom_out_factor)
        height = int(frame.shape[0] / zoom_out_factor)
        frame_zoomed = cv2.resize(frame, (width, height))

        return frame_zoomed

# Create a Streamlit app
def app():
    # Set the title of the app
    st.title("Webcam")

    try:
        # Use the webrtc_streamer function to capture video from the default webcam
        ctx = webrtc_streamer(
            key="example",
            video_transformer_factory=ZoomOutTransformer,
            async_transform=True,
        )

        # Add a button to toggle table visibility
        if ctx.video_transformer:
            if "show_table" not in st.session_state:
                st.session_state.show_table = False
            if not st.session_state.show_table:
                if st.sidebar.button("Bảng đánh giá"):
                    st.session_state.show_table = True
            
            # Display table if button is clicked
            if st.session_state.show_table:
                st.write("Bảng đánh giá")
                st.table({
                    'Họ và tên': ['Text 1', 'Text 2'],
                    'Đánh giá': ['Text 3', 'Text 4']
                })

        # Handle webcam screen display
        if ctx.video_transformer:
            frame_zoomed = ctx.video_transformer.frame_out
            st.image(frame_zoomed, channels="BGR")

    except Exception as e:
        st.error("An error occurred. Please check your input or try again later.")

if __name__ == "__main__":
    app()
