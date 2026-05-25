from ultralytics import YOLO

def main():
    # Load a fresh model weights instance
    model = YOLO("yolo11n.pt")
    
    # Train using custom hyperparameter overrides to crush the imbalance
    model.train(
        data="sliced_data.yaml",
        epochs=30,
        imgsz=640,          # Every input image is now a high-density 640x640 crop
        batch=16,
        cls=4.0,            # BUMP classification loss weight from 1.0 to 4.0 (Forces focus on calcifications)
        box=10.0,           # Boost box detection penalties
        fraction=1.0,       # Use full dataset resources
        degrees=15.0,       # Inject data augmentations to prevent overfitting
        flipud=0.5,         # Flip vertically (perfect for shifting medical scans)
        name="gcf_brca_fixed_matrix"
    )

if __name__ == "__main__":
    main()
