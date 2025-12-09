import os
import json
import shutil
from tqdm import tqdm

def prepare_docvqa_dataset(base_data_path):
    docvqa_path = os.path.join(base_data_path, "docvqa")
    images_path = os.path.join(docvqa_path, "images")
    train_images_path = os.path.join(images_path, "train")
    val_images_path = os.path.join(images_path, "val")
    annotations_path = os.path.join(docvqa_path, "annotations")

    # 1. Create necessary directories
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(annotations_path, exist_ok=True)
    print(f"Created directories: {train_images_path}, {val_images_path}, {annotations_path}")

    # 2. Move annotation files to annotations/
    annotation_files = [
        "test_v1.0.json",
        "train_v1.0_withQT.json",
        "val_v1.0_withQT.json",
    ]
    for file_name in annotation_files:
        src = os.path.join(docvqa_path, file_name)
        dst = os.path.join(annotations_path, file_name)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {file_name} to {annotations_path}")
        elif os.path.exists(dst): # Already moved
            print(f"{file_name} already in {annotations_path}")
        else:
            print(f"Warning: {file_name} not found in {docvqa_path} or {annotations_path}")

    # 3. Read annotation files and move images
    # DocVQA JSON structure: 'images' key contains list of image metadata
    # The image files are currently in docvqa_path directly.

    # Function to move images based on annotation file
    def process_images_based_on_json(json_file_path, target_image_dir):
        if not os.path.exists(json_file_path):
            print(f"Error: Annotation file not found: {json_file_path}")
            return

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # DocVQA JSON structure: 'data' key contains list of image metadata
        image_info_list = data.get('data', []) # Changed from 'images' to 'data'
        
        print(f"Processing images for {os.path.basename(json_file_path)} into {target_image_dir}...")
        for img_info in tqdm(image_info_list):
            # The 'image' key contains paths like "documents/xnbl0037_1.png"
            full_image_path_in_json = img_info['image']
            # Extract just the filename from the path
            file_name = os.path.basename(full_image_path_in_json) 
            
            src_image_path = os.path.join(docvqa_path, file_name) # Assuming images are directly in docvqa_path after tar extraction
            dst_image_path = os.path.join(target_image_dir, file_name)

            if os.path.exists(src_image_path) and not os.path.exists(dst_image_path):
                shutil.move(src_image_path, dst_image_path)
            elif os.path.exists(dst_image_path):
                # print(f"Image {file_name} already in {target_image_dir}") # Too noisy
                pass
            # else:
            #     print(f"Warning: Image {file_name} not found at {src_image_path}") # Can be noisy for test set

    process_images_based_on_json(os.path.join(annotations_path, "train_v1.0_withQT.json"), train_images_path)
    process_images_based_on_json(os.path.join(annotations_path, "val_v1.0_withQT.json"), val_images_path)
    
    # After moving train/val images, any remaining image files in docvqa_path
    # that are not spdocvqa_ocr.tar.gz can be considered test images or unclassified.
    # For now, we will leave them in docvqa_path or move them to a 'test' folder if needed.
    # For this project, we primarily focus on train/val splits for evaluation.
    
    print(f"DocVQA dataset preparation complete. Check {images_path} and {annotations_path}")


if __name__ == "__main__":
    # Assuming the script is run from donut-ocr-assisted/
    # So base_data_path should point to donut-ocr-assisted/data/
    # The current script is in donut-ocr-assisted/scripts/
    # So relative path to data is ../../data or direct path resolution
    base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    
    prepare_docvqa_dataset(base_data_path)