import streamlit as st
import cv2
import numpy as np
import time
import torch
from PIL import Image
from torchvision.transforms import functional as F
from vidgear.gears import CamGear

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Streamlit app title
st.title("YouTube Live Stream Object Detection")

# Add YouTube Video URL as input source
video_url = "https://www.youtube.com/watch?v=KyQAB-TKOVA"

# Initialize CamGear for streaming
stream = CamGear(source=video_url, backend =0,  time_delay=1, stream_mode=True).start()

# Set up Streamlit columns for layout
col1, col2 = st.columns([1, 3])

# Display YouTube live stream on the left
col1.title("Live Stream")
col1.video(video_url)

# Display captured frames with annotations on the right
col2.title("Object Detection")
latest_frame = col2.empty()

while True:
    # Read frames from the stream
    frame = stream.read()

    # Check if frame is not None
    if frame is not None:
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform object detection
        results = model(pil_image)

        # Draw annotations on the frame
        annotated_frame = results.render()[0]

        # Convert the annotated frame back to OpenCV format
        annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

        # Display the annotated frame on the right side
        latest_frame.image(annotated_frame, channels="BGR", use_column_width=True)
    
    # Delay for 0.2 seconds
    time.sleep(0.1)

# Safely close the video stream
stream.stop()
