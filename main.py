import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import requests
from flask import Flask, jsonify

# Load YOLO model (Update path if needed)
model = YOLO('best.pt')  # Make sure 'best.pt' is in the correct path

# Global variables
run_detection = False
esp32_ip = "http://192.168.1.100:5000"

# Initialize Flask app for API
app = Flask(__name__)

@app.route('/start', methods=['GET'])
def start():
    global run_detection
    run_detection = True
    return jsonify({"message": "Navigation started"}), 200

@app.route('/stop', methods=['GET'])
def stop():
    global run_detection
    run_detection = False
    return jsonify({"message": "Navigation stopped"}), 200

# Run Flask in a separate thread
def run_flask():
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

# Start Flask API in a separate thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Streamlit UI
import streamlit as st
st.title("Object Detection & Navigation Control")

# # Input for ESP32 IP Address
# esp32_ip = st.text_input("Enter ESP32 IP Address (e.g., http://192.168.1.100)", "")

# Streamlit buttons for manual control
if st.button("Start Detection"):
    run_detection = True
if st.button("Stop Detection"):
    run_detection = False

# Placeholder for video feed
video_placeholder = st.empty()

# OpenCV: Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    cap.release()
    st.stop()

# Start Object Detection Loop
while True:
    # if not run_detection:
    #     st.write("Click 'Start Detection' to begin.")
    #     continue

    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame.")
        break

    # Run YOLO object detection
    results = model.predict(source=frame, conf=0.25, show=False)

    # Find the center of the frame
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2

    nearest_object = None
    min_distance = float('inf')

    # Detect objects
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        object_center_x, object_center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate distance from the center
        distance = np.sqrt((object_center_x - center_x) ** 2 + (object_center_y - center_y) ** 2)

        if distance < min_distance:
            min_distance = distance
            nearest_object = (class_name, x1, y1, x2, y2, object_center_x, object_center_y)

    if nearest_object:
        class_name, x1, y1, x2, y2, object_center_x, object_center_y = nearest_object
        st.write(f"Nearest Object: {class_name} at {min_distance:.2f} pixels")

        # Determine movement direction
        move_command = ""
        if object_center_x < center_x - 50:
            move_command = "LEFT"
        elif object_center_x > center_x + 50:
            move_command = "RIGHT"
        elif object_center_y < center_y - 50:
            move_command = "FORWARD"
        elif object_center_y > center_y + 50:
            move_command = "BACKWARD"

        if move_command and esp32_ip:
            try:
                requests.get(f"{esp32_ip}/{move_command.lower()}")
                st.write(f"Command Sent: {move_command}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to send command: {e}")

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert frame from BGR to RGB
    annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display video feed
    video_placeholder.image(annotated_frame, channels="RGB")

cap.release()
cv2.destroyAllWindows()
