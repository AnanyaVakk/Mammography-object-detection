import cv2
import os
import glob
from ultralytics import YOLO

# Define paths
IMAGE_DIR = "/Users/ananya/Desktop/BRCA/gcf_sliced_dataset/val/images"
OUTPUT_DIR = "/Users/ananya/Desktop/BRCA/outputs/triage_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load your custom fine-tuned weights file
model = YOLO("/Users/ananya/Desktop/BRCA/runs/detect/gcf_brca_fixed_matrix/weights/best.pt")

# Grab a clean sequence of test frames
test_frames = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))[:20]

print(f"Executing batch inference on {len(test_frames)} validation scans...")

for idx, img_path in enumerate(test_frames):
    file_name = os.path.basename(img_path)
    
    # Run prediction using your optimized F1 threshold (0.27)
    results = model.predict(img_path, conf=0.27, imgsz=640, verbose=False)[0]
    
    # Render and save the annotated output frame with bounding boxes
    annotated_frame = results.plot()
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"triage_{file_name}"), annotated_frame)
    
    # Retrieve and print raw bounding box coordinates to verify task completion
    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = [round(coord, 2) for coord in box.xyxy[0].tolist()] # [xmin, ymin, xmax, ymax]
        
        class_name = "mass" if class_id == 0 else "calcification"
        print(f"Frame {idx:02d} | Detected: {class_name} | Conf: {confidence:.2f} | BBox: {xyxy}")

print(f"Inference complete! Annotated validation frames saved cleanly to {OUTPUT_DIR}")
