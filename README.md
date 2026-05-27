# Mammography-object-detection

This project is a computer vision pipeline built using Ultralytics YOLO11 for detecting breast masses and calcifications in mammography scans from the INbreast dataset.

The main goal of this project was to solve a major issue in medical object detection:

Tiny calcifications in mammograms are often lost when high-resolution medical images are resized to standard YOLO input sizes like 640×640.

Because calcifications can be only a few pixels wide, normal image downsampling causes them to disappear into the background during CNN processing.

To solve this problem, this project uses:
- CLAHE-based contrast enhancement
- XML-to-YOLO annotation conversion
- Overlapping patch tiling for high-resolution preservation
- Optimized YOLO11 training and inference

Instead of shrinking the full mammogram, large scans are sliced into overlapping 640×640 tiles so that microscopic features remain visible to the model.

This significantly improved calcification detection performance compared to standard resizing pipelines.

---

# Features

- DICOM mammogram preprocessing
- CLAHE contrast enhancement
- XML annotation parsing
- YOLO label generation
- Overlapping patch tiling
- YOLO11 training and inference
- Confidence threshold optimization

---

# Dataset

- INbreast Dataset
- Mammography DICOM images
- Expert radiologist annotations

Classes:
- `mass`
- `calcification`

---

# Pipeline

```text
DICOM Images
    ↓
CLAHE Enhancement
    ↓
XML Annotation Parsing
    ↓
YOLO Label Conversion
    ↓
Overlapping Patch Tiling
    ↓
YOLO11 Training
    ↓
Inference + Evaluation
```

---

# Results

The overlapping tiling pipeline significantly improved microscopic calcification detection compared to standard image resizing.

Key improvements:
- Better small-object preservation
- Higher calcification recall
- Improved F1-score and mAP

---

# Future Enhancements

- Multi-dataset validation (CBIS-DDSM, VinDr-Mammo)
- Segmentation-based lesion localization
- Real-time folder monitoring for automatic inference
- Improved small-object augmentation techniques

---

# Technologies Used

- Python
- PyTorch
- Ultralytics YOLO11
- OpenCV
- NumPy
- pydicom
- matplotlib
  
---

# Author

Ananya Vakkalanka
(B.Tech Biotechnology)
