import os
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import threading
import datetime
import re
import io
from PIL import Image
import gc

# Initialize YOLO model
model_path = "yolo11s.pt"  # Model in the same directory
try:
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Successfully loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Detection will be disabled.")
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Camera stream configuration
camera_stream_url = "http://demo/livestream.cgi?stream=12&action=play&media=video_audio_data"

# Alternative URL formats to try if the main one fails
alternative_urls = [
    "rtsp://demo/live",  # Try RTSP protocol
    "http://demo/video"    # Simplified HTTP path
]

# Initialize Flask app
app = Flask(__name__)

# Global variables to store detection results
detection_counts = {"Birds": 0, "Persons": 0}
last_frame = None
processing_active = True
use_default_camera = False
last_analysis_time = 0  # Track when we last analyzed a frame
frame_count_since_analysis = 0  # Track frames since last analysis
max_frames_without_analysis = 500  # Only run model every 500 frames

def filter_detections(results, classes=["person", "bird"]):
    # Get the first result's boxes
    boxes = results[0].boxes
    # Filter boxes where class is either person or bird
    mask = np.array([results[0].names[int(c)] in classes for c in boxes.cls])
    filtered_result = results[0][mask]

    # Count birds and persons
    class_counts = {"Birds": 0, "Persons": 0}
    for box in filtered_result.boxes:
        class_name = results[0].names[int(box.cls)]
        if class_name == "bird":
            class_counts["Birds"] += 1
        elif class_name == "person":
            class_counts["Persons"] += 1

    return filtered_result, class_counts

def process_camera_stream():
    global detection_counts, last_frame, processing_active, use_default_camera, last_analysis_time, frame_count_since_analysis
    
    # Try to use the IP camera first
    print(f"Attempting to connect to IP camera stream: {camera_stream_url}")
    
    # Variables for connection retry logic
    connection_retry_interval = 30  # seconds
    last_connection_attempt = 0
    connected = False
    cap = None
    
    while processing_active:
        current_time = time.time()
        
        # If not connected or if it's time to retry connection
        if not connected or (current_time - last_connection_attempt > connection_retry_interval):
            # Close any existing capture
            if cap is not None:
                cap.release()
                
            # Try to connect to the camera
            last_connection_attempt = current_time
            print(f"Attempting to connect to IP camera at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            cap = cv2.VideoCapture(camera_stream_url)
            if not cap.isOpened():
                print(f"Could not open IP camera stream at {camera_stream_url}")
                print("Error: Failed to connect to IP camera")
                
                # Try alternative URLs
                for url in alternative_urls:
                    print(f"Trying alternative URL: {url}")
                    cap = cv2.VideoCapture(url)
                    if cap.isOpened():
                        print(f"Successfully connected to IP camera stream using alternative URL: {url}")
                        connected = True
                        break
                    else:
                        print(f"Failed to connect to IP camera stream using alternative URL: {url}")
                
                if not cap.isOpened():
                    # Create a blank frame as a fallback
                    blank_frame = np.zeros((480, 640, 3), np.uint8)
                    cv2.putText(blank_frame, "Camera connection failed", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(blank_frame, f"Retrying in {connection_retry_interval}s", (50, 280), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', blank_frame)
                    last_frame = buffer.tobytes()
                    connected = False
                    time.sleep(1)  # Sleep briefly before continuing the loop
                    continue
            else:
                print("Successfully connected to IP camera stream")
                connected = True
        
        # If we're connected, try to read frames
        if connected:
            try:
                # Read frame from the video stream
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Error reading from IP camera, will retry connection")
                    connected = False
                    continue
                
                current_time = time.time()
                time_since_last_analysis = current_time - last_analysis_time
                # Process frame much less frequently (every 60 seconds or every 500 frames)
                if time_since_last_analysis >= 60 or frame_count_since_analysis >= max_frames_without_analysis:  
                    print(f"Analyzing frame at {time.strftime('%Y-%m-%d %H:%M:%S')} (Time since last analysis: {time_since_last_analysis:.1f}s, Frames since last analysis: {frame_count_since_analysis})")
                    # Perform detection on the frame
                    if model is not None:
                        results = model(frame)
                        
                        # Filter for only birds and persons
                        filtered_results, counts = filter_detections(results)
                        
                        # Update global detection counts
                        detection_counts = counts
                        
                        # Get the annotated frame
                        annotated_frame = filtered_results.plot()
                        
                        # Add count text to the frame
                        text = f"Birds: {counts['Birds']} Persons: {counts['Persons']}"
                        cv2.putText(
                            annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        )
                        
                        # Convert the frame to JPEG format for streaming
                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        last_frame = buffer.tobytes()
                        
                        # Update the last analysis time
                        last_analysis_time = current_time
                        
                        # Reset frame count
                        frame_count_since_analysis = 0
                        
                        # Log the detection counts
                        print(f"Detection counts updated: Birds: {counts['Birds']}, Persons: {counts['Persons']}")
                    else:
                        print("Model not loaded, skipping detection")
                        _, buffer = cv2.imencode('.jpg', frame)
                        last_frame = buffer.tobytes()
                        
                        # Increment frame count
                        frame_count_since_analysis += 1
                        
                        # Add debug information if frame is not being displayed
                        if ret and frame is not None:
                            # Check if frame is valid
                            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                                print("Warning: Invalid frame received (empty or zero dimension)")
                            else:
                                height, width = frame.shape[:2]
                                if frame_count_since_analysis % 100 == 0:  # Only log every 100 frames to reduce output
                                    print(f"Frame size: {width}x{height}, type: {frame.dtype}, frame count: {frame_count_since_analysis}")
                
                # Sleep a bit to reduce CPU usage but still keep video feed smooth
                time.sleep(0.05)  # Reduced from 0.1 to 0.05 for more responsive analysis
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                connected = False
                time.sleep(1)
        else:
            # Not connected, sleep before retry
            time.sleep(1)

def generate_frames():
    global last_frame
    
    while True:
        if last_frame is not None:
            try:
                # Add a small delay to control frame rate
                time.sleep(0.1)  # 10 FPS to reduce memory usage
                # Make sure we're sending valid JPEG data
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
                # Force garbage collection to prevent memory leaks
                gc.collect()
            except Exception as e:
                print(f"Error in generate_frames: {e}")
                time.sleep(0.5)
        else:
            # Generate a blank frame if no frame is available
            blank_frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_frame, "Camera Disconnected", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            # Force garbage collection
            gc.collect()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Set response headers to prevent caching and memory buildup
    response = Response(generate_frames(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    # Set headers to prevent connection issues
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Connection'] = 'keep-alive'
    return response

@app.route('/detection_data')
def detection_data():
    return jsonify(detection_counts)

@app.route('/data')
def api_data():
    global detection_counts, use_default_camera, last_frame, last_analysis_time
    
    # Get current timestamp
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate seconds since last analysis
    seconds_since_analysis = int(time.time() - last_analysis_time)
    
    # Create a comprehensive data object
    data = {
        "timestamp": current_time,
        "counts": detection_counts,
        "camera": {
            "status": "connected" if last_frame is not None else "disconnected",
            "source": "IP Camera" if not use_default_camera else "Default Camera"
        },
        "analysis": {
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_analysis_time)) if last_analysis_time > 0 else "Never",
            "seconds_since_update": seconds_since_analysis,
            "update_frequency": "60 seconds"
        }
    }
    
    return jsonify(data)

@app.route('/camera_status')
def camera_status():
    global use_default_camera
    return jsonify({
        "status": "connected" if last_frame is not None else "disconnected",
        "source": "IP Camera" if not use_default_camera else "Default Camera"
    })

def create_templates_folder():
    import os
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # Create index.html template
    index_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Stream</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .video-container {
                display: flex;
                justify-content: center;
                margin: 20px 0;
            }
            .stats {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
                flex-wrap: wrap;
            }
            .stat-card {
                background-color: #f9f9f9;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
                min-width: 200px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
                color: #2c3e50;
            }
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 5px;
            }
            .status-connected {
                background-color: #2ecc71;
            }
            .status-disconnected {
                background-color: #e74c3c;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
            }
            @media (max-width: 768px) {
                .stats {
                    flex-direction: column;
                    align-items: center;
                }
                .stat-card {
                    width: 90%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Camera Stream</h1>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Camera Stream">
            </div>
            <div class="stats">
                <div class="stat-card">
                    <h3>Birds</h3>
                    <div class="stat-value" id="bird-count">0</div>
                </div>
                <div class="stat-card">
                    <h3>Persons</h3>
                    <div class="stat-value" id="person-count">0</div>
                </div>
                <div class="stat-card">
                    <h3>Camera Status</h3>
                    <div>
                        <span class="status-indicator" id="status-indicator"></span>
                        <span id="camera-status">Unknown</span>
                    </div>
                    <div id="camera-source">Unknown</div>
                </div>
                <div class="stat-card">
                    <h3>Last Update</h3>
                    <div id="last-update">Never</div>
                    <div id="update-time-ago">N/A</div>
                </div>
            </div>
        </div>

        <script>
            // Function to update the detection counts
            function updateCounts() {
                fetch('/data')
                    .then(response => response.json())
                    .then(data => {
                        // Update bird and person counts
                        document.getElementById('bird-count').textContent = data.counts.Birds;
                        document.getElementById('person-count').textContent = data.counts.Persons;
                        
                        // Update camera status
                        const statusIndicator = document.getElementById('status-indicator');
                        const cameraStatus = document.getElementById('camera-status');
                        const cameraSource = document.getElementById('camera-source');
                        
                        if (data.camera.status === 'connected') {
                            statusIndicator.className = 'status-indicator status-connected';
                            cameraStatus.textContent = 'Connected';
                        } else {
                            statusIndicator.className = 'status-indicator status-disconnected';
                            cameraStatus.textContent = 'Disconnected';
                        }
                        
                        cameraSource.textContent = data.camera.source;
                        
                        // Update last analysis time if available
                        if (data.analysis) {
                            document.getElementById('last-update').textContent = data.analysis.last_update;
                            document.getElementById('update-time-ago').textContent = 
                                `${data.analysis.seconds_since_update} seconds ago`;
                        }
                    })
                    .catch(error => console.error('Error fetching data:', error));
            }
            
            // Update counts immediately and then every 2 seconds
            updateCounts();
            setInterval(updateCounts, 2000);
        </script>
    </body>
    </html>
    '''
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)

def main():
    # Create templates folder and index.html if they don't exist
    create_templates_folder()
    
    # Start the camera stream processing in a separate thread
    camera_thread = threading.Thread(target=process_camera_stream)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Start the Flask web server
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)

if __name__ == "__main__":
    main()
