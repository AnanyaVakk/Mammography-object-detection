import os
import glob
import re
import xml.etree.ElementTree as ET
import cv2

# Define exact paths matching your framework layout
XML_DIR = "./raw_data/AllXML/"
DATASET_DIR = "./dataset"

def parse_inbreast_xml(xml_path):
    """Parses INbreast plist XML files to extract lesion type and boundary boxes."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return []

    rois = []
    
    # Locate all dictionary definitions mapping isolated regions of interest
    dict_elements = root.findall(".//dict/dict/dict")
    if not dict_elements:
        dict_elements = root.findall(".//dict")

    for d in dict_elements:
        name_str = None
        points = []
        
        elements = list(d)
        for idx, el in enumerate(elements):
            if el.tag == 'key' and el.text == 'Name':
                if idx + 1 < len(elements) and elements[idx+1].tag == 'string':
                    # FIXED: Added a check to handle empty/None text entries safely
                    raw_text = elements[idx+1].text
                    name_str = raw_text.lower() if raw_text else "unknown"
            
            if el.tag == 'key' and el.text == 'Point_px':
                if idx + 1 < len(elements) and elements[idx+1].tag == 'array':
                    array_strings = elements[idx+1].findall("string")
                    for s in array_strings:
                        if s.text:
                            # Standardize coordinate pairs formatted like '(3187.39, 1992.23)'
                            match = re.findall(r"[-+]?\d*\.\d+|\d+", s.text)
                            if len(match) >= 2:
                                points.append((float(match[0]), float(match[1])))
        
        if name_str and points:
            if 'mass' in name_str or 'spiculated' in name_str or 'unknown' in name_str:
                class_id = 0  # Class 0: Mass (defaulting unknown shapes to mass for triage safety)
            elif 'calcification' in name_str:
                class_id = 1  # Class 1: Calcification
            else:
                continue
                
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            bbox = (class_id, min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            rois.append(bbox)
            
    return rois

def convert_to_yolo_labels(split_type):
    """Matches generated image frame directories and exports normalized label coordinates."""
    image_paths = glob.glob(os.path.join(DATASET_DIR, split_type, "images", "*.jpg"))
    labels_dir = os.path.join(DATASET_DIR, split_type, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    total_bounding_boxes = 0
    matched_files = 0

    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        
        # Extract the leading 8 digits (e.g., '20586908' from '20586908_6c613a14b80a8591_MG_R_CC_ANON.jpg')
        id_match = re.match(r'^(\d+)', file_name)
        if not id_match:
            continue
            
        file_id = id_match.group(1)
        
        # Locate corresponding XML file matching the exact folder name
        xml_path = os.path.join(XML_DIR, f"{file_id}.xml")
        label_file_path = os.path.join(labels_dir, file_name.replace(".jpg", ".txt"))
        
        # Create an empty annotation file if no XML structure is provided
        if not os.path.exists(xml_path):
            with open(label_file_path, "w") as f:
                pass
            continue
            
        # Get dimensions to normalize coordinates
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        
        rois = parse_inbreast_xml(xml_path)
        
        if rois:
            matched_files += 1
            
        with open(label_file_path, "w") as f:
            for class_id, x_min, y_min, x_max, y_max in rois:
                # Convert raw pixels to normalized YOLO bounding boxes
                x_center = ((x_min + x_max) / 2.0) / img_w
                y_center = ((y_min + y_max) / 2.0) / img_h
                width = (x_max - x_min) / img_w
                height = (y_max - y_min) / img_h
                
                # Clip coordinates to 0.0-1.0 boundary window
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                width = min(max(width, 0.0), 1.0)
                height = min(max(height, 0.0), 1.0)
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_bounding_boxes += 1

    print(f"Generated labels for {split_type} set.")
    print(f" -> Images containing visible abnormalities: {matched_files}")
    print(f" -> Total active bounding boxes drawn: {total_bounding_boxes}\n")

# Run cross-validation parsing loops
print("Converting training XML profiles...")
convert_to_yolo_labels("train")
print("Converting validation XML profiles...")
convert_to_yolo_labels("val")
print("Data conversion successfully complete!")
