import streamlit as st
import cv2
import numpy as np
import asyncio
from ultralytics import YOLO
import requests
import re

# Load YOLOv5 model
model = YOLO('yolov8n.pt')

# Streamlit app title
st.title("YouTube Live Stream Object Detection")

# Add YouTube Video URL as input source
video_url = "https://www.youtube.com/watch?v=dIChLG4_WNs"

# Set up Streamlit columns for layout
col1, col2 = st.columns([1, 3])

# Display YouTube live stream on the left
col1.title("Live Stream")
col1.video(video_url)

# Display captured frames with annotations on the right
col2.title("Object Detection")
latest_frame = col2.empty()

# Function to fetch video stream URL from YouTube video page
def fetch_video_stream_url(youtube_url):
    try:
        # Send GET request to YouTube video page
        response = requests.get(youtube_url)

        # Extract video stream URL using regular expression
        pattern = r'"hlsManifestUrl":"(.*?)"'
        match = re.search(pattern, response.text)
        if match:
            video_stream_url = match.group(1)
            return video_stream_url
        else:
            print("Error: Video stream URL not found")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Asynchronous function to capture frame using OpenCV
async def capture_frame(video_stream_url):
    try:
        # Open video stream using OpenCV
        cap = cv2.VideoCapture(video_stream_url)

        while True:
            # Read the frame
            ret, frame = cap.read()
            
            # If frame is read successfully, yield it
            if ret:
                yield frame
            else:
                print("Error: Failed to read frame")
                break
    except Exception as e:
        print(f"Error: {e}")

# Asynchronous function to perform object detection
async def detect_objects():
    url = fetch_video_stream_url(video_url)
    async for frame in capture_frame(url):
        # Perform object detection
        frame_resized = cv2.resize(frame, (640, 480))
        results = model.predict(source=frame_resized)
        res_plotted = results[0].plot()
        
        # Display the annotated frame on the right side
        latest_frame.image(res_plotted, use_column_width=True)
        

# Run the main coroutine
async def main():
    await asyncio.gather(
        detect_objects()
    )

# Run the main coroutine
asyncio.run(main())