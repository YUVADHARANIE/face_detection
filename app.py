import cv2
import tempfile
import streamlit as st
import numpy as np
import os

# Function to detect faces in the video
def detect_faces_in_video(video_path, cascade_path):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Create a temporary file to save the processed video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file_path = temp_file.name
    temp_file.close()
    
    video_writer = cv2.VideoWriter(temp_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        video_writer.write(frame)
    
    cap.release()
    video_writer.release()
    
    return temp_file_path

# Streamlit App Interface
st.title('Face Detection in Video')

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.text("Processing video...")
    
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_video.read())
        tfile_path = tfile.name
    
    # Detect faces in the video
    cascade_src = 'haarcascade_frontalface_default.xml'  # Path to the Haar Cascade for face detection
    processed_video_path = detect_faces_in_video(tfile_path, cascade_src)
    
    # Provide a download link for the processed video
    st.text("Face detection completed. You can download the processed video using the link below:")
    with open(processed_video_path, "rb") as f:
        st.download_button(
            label="Download Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
