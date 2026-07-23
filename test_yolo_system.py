from ultralytics import YOLO
import cv2

def main():
    # --- 1. CONFIGURATION ---
    # Point this to your newly trained weights!
    model_path = model_path = r"C:\Users\sacha\runs\detect\Traffic_Models\yolo_run_12\weights\best.pt"
    
    # Point this to a test video in your folder
    video_path = r"traffic_videos\v2.mp4"

    print("Loading custom YOLO model...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model. Make sure '{model_path}' exists!")
        return

    # --- 2. START VIDEO ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video '{video_path}'. Check the filename!")
        return

    print("✅ System Ready. Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # --- 3. RUN INFERENCE ---
        # YOLOv8 makes this incredibly simple
        results = model(frame, verbose=False)[0]

        vehicle_count = 0
        has_ambulance = False

        # --- 4. PARSE RESULTS & DRAW BOXES ---
        for box in results.boxes:
            conf = float(box.conf[0])
            
            # Only count vehicles the model is at least 50% sure about
            if conf > 0.50: 
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class ID
                cls_id = int(box.cls[0])
                
                # THE UPGRADE: Automatically pull the exact name from the model's memory
                label_name = model.names[cls_id]

                # Priority & Color Logic
                if label_name == 'ambulance':
                    has_ambulance = True
                    color = (0, 0, 255)       # Red for ambulance
                elif label_name == 'truck':
                    color = (0, 165, 255)     # Orange for truck (OpenCV uses BGR, so 0,165,255)
                else:
                    color = (0, 255, 0)       # Green for cars, buses, and bikes

                # Draw the box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label_name} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                vehicle_count += 1

        # --- 5. DASHBOARD VISUALIZATION ---
        # Create a black background for the text
        cv2.rectangle(frame, (0, 0), (350, 100), (0, 0, 0), -1)
        
        if has_ambulance:
            cv2.putText(frame, "STATUS: EMERGENCY OVERRIDE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "STATUS: NORMAL TRAFFIC", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the video
        cv2.imshow("Smart Traffic System - YOLOv8", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()