import os
import glob
import cv2
import numpy as np

def slice_meta_dataset(split):
    img_paths = glob.glob(f"./dataset/{split}/images/*.jpg")
    
    # New output directories for our high-resolution sliced dataset
    out_img_dir = f"./gcf_sliced_dataset/{split}/images"
    out_lab_dir = f"./gcf_sliced_dataset/{split}/labels"
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lab_dir, exist_ok=True)
    
    tile_size = 640
    stride = 512 # 128 pixels of overlap so we don't cut a tumor in half completely
    tile_count = 0

    for img_path in img_paths:
        base_name = os.path.basename(img_path).replace(".jpg", "")
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # Read matching label file
        label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # Convert normalized back to raw pixel coordinates
                        cid, cx, cy, bw, bh = map(float, parts)
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        x2 = (cx + bw/2) * w
                        y2 = (cy + bh/2) * h
                        bboxes.append([int(cid), x1, y1, x2, y2])

        # Slide a 640x640 window across the entire mammogram
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = img[y:y+tile_size, x:x+tile_size]
                
                # Check which bounding boxes fall inside this specific tile
                tile_bboxes = []
                for cid, x1, y1, x2, y2 in bboxes:
                    # Calculate intersection box
                    ix1 = max(x1, x)
                    iy1 = max(y1, y)
                    ix2 = min(x2, x + tile_size)
                    iy2 = min(y2, y + tile_size)
                    
                    # If the anomaly is inside this tile, recalculate local coordinates
                    if ix2 > ix1 and iy2 > iy1:
                        # Convert to local tile pixel coordinates
                        tx1 = ix1 - x
                        ty1 = iy1 - y
                        tx2 = ix2 - x
                        ty2 = iy2 - y
                        
                        # Convert back to normalized YOLO format (relative to 640x640 tile)
                        tcx = ((tx1 + tx2) / 2.0) / tile_size
                        tcy = ((ty1 + ty2) / 2.0) / tile_size
                        tbw = (tx2 - tx1) / tile_size
                        tbh = (ty2 - ty1) / tile_size
                        
                        tile_bboxes.append(f"{cid} {tcx:.6f} {tcy:.6f} {tbw:.6f} {tbh:.6f}")
                
                # To keep data balanced, save ALL tiles with abnormalities, 
                # but only 10% of empty background tiles
                if len(tile_bboxes) > 0 or np.random.rand() < 0.10:
                    tile_name = f"{base_name}_tile_{y}_{x}.jpg"
                    cv2.imwrite(os.path.join(out_img_dir, tile_name), tile)
                    
                    with open(os.path.join(out_lab_dir, tile_name.replace(".jpg", ".txt")), "w") as f:
                        for bbox_str in tile_bboxes:
                            f.write(bbox_str + "\n")
                    tile_count += 1
                    
    print(f"Generated {tile_count} optimized sliced frames for the {split} configuration.")

print("Slicing training set...")
slice_meta_dataset("train")
print("Slicing validation set...")
slice_meta_dataset("val")
print("Data scaling complete!")
