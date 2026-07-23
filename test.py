import os
import random
import shutil

def split_dataset(source_dir, test_dir, split_ratio=0.3):

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created directory: {test_dir}")

   
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_files = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(valid_extensions)]
    
    random.shuffle(all_files)
    num_test_files = int(len(all_files) * split_ratio)
    
    test_files = all_files[:num_test_files]

    print(f"Moving {len(test_files)} images to {test_dir}...")

 
    for file_name in test_files:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(test_dir, file_name)
       
        shutil.move(source_path, destination_path)

    print("Task complete!")

# --- Configuration ---
SOURCE_PATH = r'C:\Users\sacha\OneDrive\Desktop\Smart traffic management system\new_train_images'
TEST_PATH = r'C:\Users\sacha\OneDrive\Desktop\Smart traffic management system\testing_folder'

split_dataset(SOURCE_PATH, TEST_PATH)
