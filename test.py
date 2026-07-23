import os
import random
import shutil

def split_dataset(source_dir, test_dir, split_ratio=0.3):
    # 1. Create the testing directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created directory: {test_dir}")

    # 2. Get a list of all files in the source folder
    # This filters for common image extensions to avoid grabbing system files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_files = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(valid_extensions)]
    
    # 3. Shuffle and calculate the number of files to move
    random.shuffle(all_files)
    num_test_files = int(len(all_files) * split_ratio)
    
    test_files = all_files[:num_test_files]

    print(f"Moving {len(test_files)} images to {test_dir}...")

    # 4. Move the files
    for file_name in test_files:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(test_dir, file_name)
        
        # Use shutil.move to transfer them, or shutil.copy to keep originals
        shutil.move(source_path, destination_path)

    print("Task complete!")

# --- Configuration ---
SOURCE_PATH = r'C:\Users\sacha\OneDrive\Desktop\Smart traffic management system\new_train_images'
TEST_PATH = r'C:\Users\sacha\OneDrive\Desktop\Smart traffic management system\testing_folder'

split_dataset(SOURCE_PATH, TEST_PATH)