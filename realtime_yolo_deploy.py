from ultralytics import YOLO
import cv2
import serial # NEW: Import the serial library
import time

def main():
    # --- 1. CONFIGURATION ---
    model_path = r"C:\Users\sacha\runs\detect\Traffic_Models\yolo_run_12\weights\best.pt"
    
    # NEW: Connect to Arduino
    arduino_port = 'COM9' 
    
    print(f"Connecting to Arduino on {arduino_port}...")
    try:
        arduino = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2) # Give Arduino a second to reboot after connecting
        print("✅ Arduino Connected!")
    except Exception as e:
        print(f"❌ Error connecting to Arduino. Is it plugged in? Is the port correct?")
        return

    print("Loading custom YOLOv8 Edge Model...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- 2. START LIVE WEBCAM ---
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✅ System Ready. Press 'q' to stop.")

    # A variable to prevent spamming the Arduino with the same command 30 times a second
    last_command = '' 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 3. RUN INFERENCE ---
        results = model(frame, verbose=False)[0]

        vehicle_count = 0
        has_ambulance = False

        # --- 4. PARSE RESULTS ---
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

        # --- 5. HARDWARE LOGIC ---
       # --- 5. LOGIC & VIRTUAL LIGHTS ---
        cv2.rectangle(frame, (0, 0), (450, 100), (0, 0, 0), -1)
        
        # Draw the base of the virtual traffic light
        cv2.circle(frame, (400, 50), 30, (50, 50, 50), -1) 

        if has_ambulance:
            cv2.putText(frame, "STATUS: EMERGENCY OVERRIDE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (400, 50), 30, (0, 255, 0), -1) # Draw GREEN Circle
        elif vehicle_count > 5:
            cv2.putText(frame, "STATUS: HIGH DENSITY", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.circle(frame, (400, 50), 30, (0, 255, 0), -1) # Draw GREEN Circle
        else:
            cv2.putText(frame, "STATUS: NORMAL CYCLE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (400, 50), 30, (0, 0, 255), -1) # Draw RED Circle
            
        cv2.putText(frame, f"Live Vehicles Detected: {vehicle_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Send the command to the Arduino ONLY if it changed
        if current_command != last_command:
            arduino.write(current_command.encode()) # Convert string to bytes and send
            last_command = current_command

        cv2.imshow("Smart Traffic Intersection - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    arduino.close() # Clean up the serial port when quitting

if __name__ == '__main__':
    main()