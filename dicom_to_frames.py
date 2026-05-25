import os
import glob
import pydicom
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Define paths
DICOM_DIR = "./raw_data/AllDICOMs/"
OUTPUT_DIR = "./dataset/"

# 2. Get all DICOM files and split them 80% train, 20% val
dcm_files = glob.glob(os.path.join(DICOM_DIR, "*.dcm"))
train_files, val_files = train_test_split(dcm_files, test_size=0.2, random_state=42)

def process_and_save(file_list, split_type):
    for dcm_path in file_list:
        file_id = os.path.basename(dcm_path).replace(".dcm", "")
        
        # Read medical image
        ds = pydicom.dcmread(dcm_path)
        img_array = ds.pixel_array.astype(float)
        
        # Normalize to standard 0-255 grayscale
        img_array = (img_array / img_array.max()) * 255.0
        img_8bit = np.uint8(img_array)
        
        # Fix inverted images (if background is accidentally white)
        if np.mean(img_8bit) > 127:
            img_8bit = cv2.bitwise_not(img_8bit)
            
        # Apply CLAHE contrast enhancement to sharpen tumor boundaries
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_img = clahe.apply(img_8bit)
        
        # Save into the corresponding YOLO folder
        save_path = os.path.join(OUTPUT_DIR, split_type, "images", f"{file_id}.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, enhanced_img)

# 3. Execute the extraction
print("Extracting train frames...")
process_and_save(train_files, "train")
print("Extracting validation frames...")
process_and_save(val_files, "val")
print("Frame extraction complete!")
