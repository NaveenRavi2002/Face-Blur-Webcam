import cv2
import gradio as gr
import numpy as np
from threading import Thread
import time

class WebcamController:
    def __init__(self):
        self.video = None
        self.running = False
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    
    def start_webcam(self):
        """Start the webcam and face detection"""
        if self.running:
            return "Webcam is already running!"
        
        self.video = cv2.VideoCapture(0)  # Use 0 for default camera
        if not self.video.isOpened():
            return "Error: Could not open webcam"
        
        self.running = True
        return "Webcam started successfully!"
    
    def stop_webcam(self):
        """Stop the webcam"""
        if not self.running:
            return "Webcam is not running!"
        
        self.running = False
        if self.video:
            self.video.release()
            self.video = None
        return "Webcam stopped successfully!"
    
    def get_frame(self):
        """Get frame from webcam with face blur applied"""
        if not self.running or self.video is None:
            # Return a blank frame when not running
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera Not Active", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return blank
        
        ret, frame = self.video.read()
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )
        
        # Apply blur to detected faces
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame[y:y+h, x:x+w] = cv2.medianBlur(frame[y:y+h, x:x+w], 35)
        
        # Convert BGR to RGB for Gradio
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

# Initialize controller
controller = WebcamController()

def stream_webcam():
    """Generator function to stream webcam frames"""
    while True:
        frame = controller.get_frame()
        yield frame
        time.sleep(0.03)  # ~30 FPS

# Create Gradio interface
with gr.Blocks(title="Face Blur Webcam") as demo:
    gr.Markdown("# 🎥 Face Blur Webcam")
    gr.Markdown("Start the webcam to see live video with face detection and blur effect!")
    
    with gr.Row():
        start_btn = gr.Button("▶️ Start Webcam", variant="primary")
        stop_btn = gr.Button("⏹️ Stop Webcam", variant="stop")
    
    status_text = gr.Textbox(label="Status", value="Ready to start", interactive=False)
    
    video_output = gr.Image(label="Live Camera Feed", streaming=True)
    
    # Button actions
    start_btn.click(
        fn=controller.start_webcam,
        outputs=status_text
    ).then(
        fn=stream_webcam,
        outputs=video_output
    )
    
    stop_btn.click(
        fn=controller.stop_webcam,
        outputs=status_text
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()