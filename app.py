from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


model_path = r"C:\Users\sacha\runs\detect\Traffic_Models\yolo_run_12\weights\best.pt"
model = YOLO(model_path)
print("✅ YOLO Model Loaded for Simulation!")


active_sources = {
    'feed1': 0, 
    'feed2': None 
}

def generate_frames(feed_id):
    video_source = active_sources[feed_id]
    
 
    if video_source is None:
        while True:
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "AWAITING VIDEO FEED...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    cap = cv2.VideoCapture(video_source)

    while True:
        success, frame = cap.read()
        if not success:
            break

        
        results = model(frame, verbose=False)[0]
        vehicle_count = 0
        has_ambulance = False

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > 0.50: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_name = model.names[int(box.cls[0])]

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

       
        cv2.rectangle(frame, (0, 0), (450, 100), (0, 0, 0), -1)

        if has_ambulance:
            cv2.putText(frame, "SIMULATION: EMERGENCY OVERRIDE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (400, 50), 30, (0, 255, 0), -1)
        elif vehicle_count > 5:
            cv2.putText(frame, "SIMULATION: HIGH DENSITY", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.circle(frame, (400, 50), 30, (0, 255, 0), -1) 
        else:
            cv2.putText(frame, "SIMULATION: NORMAL CYCLE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (400, 50), 30, (0, 0, 255), -1) 
            
        cv2.putText(frame, f"Live Vehicles Detected: {vehicle_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<feed_id>')
def video_feed(feed_id):
    return Response(generate_frames(feed_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload/<feed_id>', methods=['POST'])
def upload_video(feed_id):
    if 'file' not in request.files: return redirect(url_for('index'))
    file = request.files['file']
    if file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        active_sources[feed_id] = filepath 
    return redirect(url_for('index'))

@app.route('/set_live/<feed_id>/<int:cam_index>')
def set_live(feed_id, cam_index):
    active_sources[feed_id] = cam_index 
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
