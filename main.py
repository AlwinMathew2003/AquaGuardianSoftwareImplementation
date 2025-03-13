import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time
import requests
from flask import Flask, jsonify

# Load YOLO model (Update path if needed)
model = YOLO('best.pt')  # Ensure best.pt is in the correct path

# Global Variables
run_detection = False
esp32_ip = "http://192.168.1.100:5000"

# Initialize Flask app for API
app = Flask(__name__)

@app.route('/on', methods=['GET'])
def start_detection():
    global run_detection
    run_detection = True
    return jsonify({"message": "Detection started"}), 200

@app.route('/off', methods=['GET'])
def stop():
    global run_detection
    run_detection = False
    return jsonify({"message": "Detection stopped"}), 200

def run_flask():
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Streamlit UI
st.title("Object Detection with Start/Stop API")
video_placeholder = st.empty()

# OpenCV: Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not access webcam.")
    st.stop()

# Streaming Video with Object Detection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame.")
        break

    # Convert frame to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if run_detection:
        results = model.predict(source=frame_rgb, conf=0.25, show=False)

        # Find center of the frame
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2

        nearest_object = None
        min_distance = float('inf')

        # Process detected objects
        for box in results[0].boxes:
            class_id = int(box.cls[0])  # Change to `cls` from `xyxy`
            class_name = results[0].names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_center_x, obj_center_y = (x1 + x2) // 2, (y1 + y2) // 2
            distance = np.sqrt((obj_center_x - center_x) ** 2 + (obj_center_y - center_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                nearest_object = (class_name, x1, y1, x2, y2, obj_center_x, obj_center_y)

        if nearest_object:
            class_name, x1, y1, x2, y2, obj_center_x, obj_center_y = nearest_object
            st.write(f"Nearest Object: {class_name} at {min_distance:.2f} pixels")

            # Determine movement direction
            move_command = ""
            if obj_center_x < center_x - 50:
                move_command = "left"
            elif obj_center_x > center_x + 50:
                move_command = "right"
            elif obj_center_y < center_y - 50:
                move_command = "forward"
            elif obj_center_y > center_y + 50:
                move_command = "backward"

            if move_command and esp32_ip:
                try:
                    requests.get(f"{esp32_ip}/{move_command}")
                    st.write(f"Command Sent: {move_command}")
                except requests.exceptions.RequestException:
                    st.error("Failed to send command!")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert to RGB for Streamlit display
    annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(annotated_frame, channels="RGB")

    # Add a delay to reduce CPU usage
    time.sleep(2)

cap.release()
cv2.destroyAllWindows()
