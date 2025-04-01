import cv2
import numpy as np
import subprocess
import time
import torch
import torchvision
import os
import datetime
import threading
from flask import Flask, Response, render_template_string
import base64
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask app for web streaming
app = Flask(__name__)

# Global frame to be shared with Flask
global_frame = None
is_processing = True

class LiveHumanDetector:
    def __init__(self, video_url, confidence_threshold=0.5, save_output=True):
        self.video_url = video_url
        self.confidence_threshold = confidence_threshold
        self.save_output = save_output
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # COCO dataset class names (we're only interested in people - class 1)
        self.classes = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
    def get_stream_url(self):
        """Get the actual stream URL using yt-dlp"""
        command = [
            'yt-dlp', 
            '-g',  # Print URL of the video
            '-f', 'best',  # Best quality
            self.video_url
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout.strip()
        
    def detect_humans(self, frame):
        """Detect humans in a frame"""
        # Convert to RGB (PyTorch models expect RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(rgb_frame.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(img_tensor)
            
        # Process results
        boxes = predictions[0]['boxes'].cpu().numpy().astype(np.int32)
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter humans (class 1 in COCO) with confidence above threshold
        human_boxes = []
        for i, label in enumerate(labels):
            if label == 1 and scores[i] > self.confidence_threshold:  # 1 is the person class
                human_boxes.append(boxes[i])
                
        return human_boxes
        
    def draw_boxes(self, frame, boxes, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes on the frame"""
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame
    
    def run(self):
        """Main loop to process the video stream"""
        global global_frame, is_processing
        
        # Get the actual stream URL
        stream_url = self.get_stream_url()
        
        # Open the stream
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print(f"Error: Could not open video stream.")
            return
            
        print("Starting human detection.")
        
        # Create output video writer if save_output is True
        out = None
        if self.save_output:
            # Get stream dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create output directory if it doesn't exist
            os.makedirs('output', exist_ok=True)
            
            # Create output filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output/human_detection_{timestamp}.mp4"
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
            print(f"Saving output to {output_filename}")
            
        frame_count = 0
        start_time = time.time()
        
        while is_processing:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to receive frame. Stream may have ended.")
                break
                
            # Process every 3rd frame to improve performance
            if frame_count % 3 == 0:
                # Detect humans
                human_boxes = self.detect_humans(frame)
                
                # Draw boxes around humans
                frame_with_boxes = self.draw_boxes(frame, human_boxes)
                
                # Calculate and display FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Update global frame for web display
                global_frame = frame_with_boxes.copy()
                
                # Save frame to video file if enabled
                if self.save_output and out is not None:
                    out.write(frame_with_boxes)
                    
                # Print detection info
                if human_boxes:
                    print(f"Detected {len(human_boxes)} humans in frame {frame_count}")
            
            frame_count += 1
                
        # Clean up
        cap.release()
        if out is not None:
            out.close()
        print("Processing complete.")

def generate_frames():
    """Generate frames for the web stream"""
    global global_frame
    while True:
        if global_frame is not None:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', global_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in proper format for HTTP streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, yield a placeholder
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('static/placeholder.jpg', 'rb').read() + b'\r\n')
        time.sleep(0.1)  # Small delay to control frame rate

@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Human Detection Stream</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: #f0f0f0;
            }
            h1 {
                margin: 20px 0;
                color: #333;
            }
            .video-container {
                width: 80%;
                max-width: 1200px;
                margin: 0 auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            img {
                width: 100%;
                height: auto;
                display: block;
            }
        </style>
    </head>
    <body>
        <h1>Live Human Detection</h1>
        <div class="video-container">
            <img src="/video_feed" alt="Live Stream" />
        </div>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    """Route for streaming video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_web_server():
    """Start the Flask web server"""
    app.run(host='0.0.0.0', port=5005, threaded=True)

def ensure_static_dir():
    """Ensure static directory exists and create placeholder image"""
    os.makedirs('static', exist_ok=True)
    if not os.path.exists('static/placeholder.jpg'):
        # Create a simple black image as placeholder
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for stream...", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('static/placeholder.jpg', placeholder)
    
if __name__ == "__main__":
    # Create static directory and placeholder
    ensure_static_dir()
    
    # YouTube Live URL from environment variable, with fallback
    youtube_url = os.getenv('YOUTUBE_URL')
    if not youtube_url:
        raise ValueError("YOUTUBE_URL environment variable is not set. Please set it in the .env file.")

    print(f"Using YouTube URL: {youtube_url}")
    
    # Start web server in a separate thread
    web_thread = threading.Thread(target=start_web_server)
    web_thread.daemon = True
    web_thread.start()
    
    print("Web server started at http://localhost:5005")
    
    try:
        # Start the detector
        detector = LiveHumanDetector(youtube_url, save_output=True)
        detector.run()
    except KeyboardInterrupt:
        # Handle graceful shutdown
        print("Shutting down...")
        is_processing = False
