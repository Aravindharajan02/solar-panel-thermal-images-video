import os
import cv2
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
from tracker import *
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO("best.pt")
class_list = model.names  # Class names from YOLO model
tracker = Tracker()  # Initialize object tracker

# Video processing function
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video settings
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{int(time.time())}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Fix: More reliable codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Panel tracking
    defective_panels = set()
    non_defective_panels = set()
    tracking_line_x = 260  
    offset = 5  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break  # Stop processing if no frames are left

        frame = cv2.resize(frame, (frame_width, frame_height))  # Fix: Ensure correct frame size

        # Run YOLO detection
        results = model.predict(frame)
        detections = results[0].boxes.data
        detections = detections.detach().cpu().numpy()
        px = pd.DataFrame(detections).astype("float")

        detected_panels = []

        for _, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            class_index = int(row[5])

            if class_index not in [0, 2]:  # Ignore unwanted classes
                continue

            class_name = class_list[class_index]
            detected_panels.append([x1, y1, x2, y2, class_name])

        # Update tracker
        bbox_id = tracker.update([box[:4] for box in detected_panels])

        # Draw detections
        for i, (x3, y3, x4, y4, obj_id) in enumerate(bbox_id):
            if i < len(detected_panels):
                class_name = detected_panels[i][4]

                # Draw bounding boxes
                color = (0, 255, 0) if class_name == "No-Anomaly" else (0, 0, 255)
                cv2.rectangle(frame, (x3, y3), (x4, y4), color, 2)
                cv2.putText(frame, class_name, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Count defective and non-defective panels
                if class_name == "defected":
                    defective_panels.add(obj_id)
                else:
                    non_defective_panels.add(obj_id)

        # Draw tracking line
        cv2.line(frame, (tracking_line_x, 50), (tracking_line_x, 450), (255, 255, 255), 3)
        cv2.putText(frame, f'Defective: {len(defective_panels)}', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f'Non-Defective: {len(non_defective_panels)}', (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)  # Fix: Ensure the frame is written to the output video

    cap.release()
    out.release()

    return output_path  # Return the correct output video path


# Homepage
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Process the uploaded video
    processed_video_path = process_video(file_path)

    return f"""
    <h2>Processing Complete!</h2>
    <a href="/download/{os.path.basename(processed_video_path)}">Download Processed Video</a>
    """
@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
