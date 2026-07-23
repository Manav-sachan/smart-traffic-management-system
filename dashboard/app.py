from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import serial
import time
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 1. INITIALIZE HARDWARE & AI (Runs once when server starts) ---
arduino_port = 'COM9'
try:
    arduino = serial.Serial(arduino_port, 9600, timeout=1)
    time.sleep(2)
    print("✅ Arduino Connected!")
except Exception as e:
    arduino = None
    print(f"⚠️ Arduino not connected: {e}. Running in software-only mode.")

model_path = r"C:\Users\sacha\runs\detect\Traffic_Models\yolo_run_12\weights\best.pt"
model = YOLO(model_path)
print("✅ YOLO Model Loaded!")

last_command = ''

# --- 2. VIDEO PROCESSING GENERATOR ---
def generate_frames(video_source):
    global last_command
    cap = cv2.VideoCapture(video_source)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO Inference
        results = model(frame, verbose=False)[0]
        vehicle_count = 0
        has_ambulance = False

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > 0.50: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label_name = model.names[cls_id]

                if label_name == 'ambulance':
                    has_ambulance = True
                    color = (0, 0, 255) 
                elif label_name == 'truck':
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0) 

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                vehicle_count += 1

        # Hardware Logic & UI Overlay
        cv2.rectangle(frame, (0, 0), (400, 100), (0, 0, 0), -1)
        current_command = ''

        if has_ambulance:
            cv2.putText(frame, "HARDWARE: EMERGENCY OVERRIDE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            current_command = 'E'
        elif vehicle_count > 5:
            cv2.putText(frame, "HARDWARE: HIGH DENSITY", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            current_command = 'H'
        else:
            cv2.putText(frame, "HARDWARE: NORMAL CYCLE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            current_command = 'N'
            
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Send to Arduino if connected
        if arduino and current_command != last_command:
            arduino.write(current_command.encode())
            last_command = current_command

        # Encode frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- 3. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/live')
def video_feed_live():
    # '0' starts the webcam
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/file/<filename>')
def video_feed_file(filename):
    # Streams the uploaded video
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Re-render the page, telling it to play the uploaded file
        return render_template('index.html', uploaded_video=filename)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)