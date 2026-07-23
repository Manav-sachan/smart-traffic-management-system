import cv2
import os


output_folder = 'new_train_images'
videos_folder = 'traffic_videos' 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


if not os.path.exists(videos_folder):
    os.makedirs(videos_folder)
    print(f"❌ I couldn't find the '{videos_folder}' folder.")
    print(f"✅ I just created it for you! Please drag your videos into it and run this script again.")
    exit()


video_files = [f for f in os.listdir(videos_folder) if f.endswith('.mp4')]

if not video_files:
    print(f"❌ The '{videos_folder}' folder is empty. Please add some .mp4 videos and try again.")
    exit()


global_saved_count = 0

print(f"Found {len(video_files)} videos. Starting extraction...")
print("-" * 50)

for video_file in video_files:
    video_path = os.path.join(videos_folder, video_file)
    print(f"Processing: {video_file}...")
    
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break 

      
        if frame_id % 30 == 0:
            filename = f"{output_folder}/frame_{global_saved_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            global_saved_count += 1
        
        frame_id += 1
        
    cap.release()

print("-" * 50)
print(f"✅ Extraction complete! Saved {global_saved_count} total images.")
