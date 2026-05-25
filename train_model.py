import os
from ultralytics import YOLO

def main():
    # 1. Initialize a lightweight pre-trained YOLO11 Nano model
    # This architecture runs incredibly fast on laptop hardware
    model = YOLO("yolo11n.pt")

    # 2. Kick off the training loop
    print("Initializing model training on INbreast dataset...")
    results = model.train(
        data="data.yaml",
        epochs=30,           # 30 epochs is a great baseline to check performance
        imgsz=640,           # Standard medical imaging resolution
        batch=16,            # Number of images processed per step
        workers=2,           # Keeps data loading smooth on consumer hardware
        name="gcf_brca_triage" # Name of your output directory
    )
    print("Training run successfully finished!")

if __name__ == "__main__":
    main()
