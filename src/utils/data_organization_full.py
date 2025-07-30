"""
Enhanced Data Organization Script - Malaria Detection Thesis
Uses both Polygon Set and Point Set data for comprehensive dataset preparation

This script:
1. Indexes all patients from both Polygon Set and Point Set
2. Creates train/validation/test splits ensuring no patient overlap
3. Generates YOLO format annotations for both polygon and point annotations
4. Creates comprehensive dataset statistics
5. Prepares data for both YOLOv11 training and CNN ensemble training

Author: Shawn Jurgen Mayol
Date: July 2025
"""

import os
import json
import yaml
import shutil
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

def create_directory_structure(base_path):
    """Create the directory structure for processed data"""
    
    directories = [
        "full_slides/train",
        "full_slides/val", 
        "full_slides/test",
        "yolo_annotations/images/train",
        "yolo_annotations/images/val",
        "yolo_annotations/images/test",
        "yolo_annotations/labels/train",
        "yolo_annotations/labels/val", 
        "yolo_annotations/labels/test",
        "extracted_cells/parasitized",
        "extracted_cells/uninfected"
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def parse_annotation_file(annotation_path):
    """Parse annotation file and return structured cell information"""
    
    if not annotation_path.exists():
        return None
    
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
        
        # Parse header: total_cells, image_width, image_height
        header = lines[0].strip().split(',')
        if len(header) >= 3:
            total_cells = int(header[0])
            img_width = int(header[1])
            img_height = int(header[2])
        else:
            return None
        
        # Parse cell annotations
        cells = []
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                cell_info = {
                    'patient_cell_id': parts[0],
                    'cell_type': parts[1],
                    'comment': parts[2],
                    'annotation_type': parts[3],
                    'num_points': int(parts[4]) if parts[4].isdigit() else 0
                }
                
                # Parse coordinates based on annotation type
                if cell_info['annotation_type'] == 'Polygon' and len(parts) >= 6:
                    coordinates = []
                    for i in range(5, len(parts), 2):
                        if i + 1 < len(parts):
                            try:
                                x = float(parts[i])
                                y = float(parts[i + 1])
                                coordinates.append((x, y))
                            except ValueError:
                                break
                    cell_info['coordinates'] = coordinates
                
                elif cell_info['annotation_type'] == 'Point' and len(parts) >= 7:
                    try:
                        x = float(parts[5])
                        y = float(parts[6])
                        cell_info['coordinates'] = [(x, y)]
                    except ValueError:
                        cell_info['coordinates'] = []
                else:
                    cell_info['coordinates'] = []
                
                cells.append(cell_info)
        
        return {
            'total_cells': total_cells,
            'img_width': img_width,
            'img_height': img_height,
            'cells': cells
        }
    
    except Exception as e:
        print(f"Error parsing {annotation_path}: {e}")
        return None

def index_all_patients(raw_uncropped_path):
    """Index all patients from both Polygon Set and Point Set"""
    
    print("🔍 Indexing all patients from both annotation sets...")
    
    patients = {}
    images = {}
    total_stats = {
        'total_parasitized': 0,
        'total_uninfected': 0,
        'total_wbc': 0,
        'total_other': 0
    }
    
    # Process both Polygon Set and Point Set
    for set_type in ['Polygon Set', 'Point Set']:
        set_path = raw_uncropped_path / set_type
        
        if not set_path.exists():
            print(f"⚠️  {set_type} not found at {set_path}")
            continue
            
        print(f"\n📂 Processing {set_type}...")
        patient_dirs = [d for d in set_path.iterdir() if d.is_dir()]
        
        for patient_dir in tqdm(patient_dirs, desc=f"Indexing {set_type}"):
            patient_id = patient_dir.name
            img_dir = patient_dir / "Img"
            gt_dir = patient_dir / "GT"
            
            if not img_dir.exists() or not gt_dir.exists():
                print(f"⚠️  Missing Img or GT directory for patient {patient_id}")
                continue
            
            # Initialize patient data
            if patient_id not in patients:
                patients[patient_id] = {
                    'id': patient_id,
                    'annotation_type': set_type.split()[0],  # 'Polygon' or 'Point'
                    'images': [],
                    'total_parasitized': 0,
                    'total_uninfected': 0,
                    'total_wbc': 0,
                    'total_other': 0
                }
            
            # Process all images for this patient
            image_files = list(img_dir.glob("*.jpg"))
            
            for img_file in image_files:
                annotation_file = gt_dir / (img_file.stem + ".txt")
                
                if not annotation_file.exists():
                    print(f"⚠️  Missing annotation for {img_file.name}")
                    continue
                
                # Parse annotations
                annotation_data = parse_annotation_file(annotation_file)
                if not annotation_data:
                    print(f"⚠️  Could not parse annotations for {img_file.name}")
                    continue
                
                # Count cell types
                cell_counts = {
                    'Parasitized': 0,
                    'Uninfected': 0,
                    'White_Blood_Cell': 0,
                    'Other': 0
                }
                
                for cell in annotation_data['cells']:
                    cell_type = cell['cell_type']
                    if cell_type in cell_counts:
                        cell_counts[cell_type] += 1
                    else:
                        cell_counts['Other'] += 1
                
                # Create image entry
                image_info = {
                    'filename': img_file.name,
                    'image_path': str(Path(set_type) / patient_id / "Img" / img_file.name),
                    'annotation_path': str(Path(set_type) / patient_id / "GT" / annotation_file.name),
                    'patient_id': patient_id,
                    'annotation_type': set_type.split()[0],
                    'cell_counts': cell_counts,
                    'image_width': annotation_data['img_width'],
                    'image_height': annotation_data['img_height'],
                    'total_cells': annotation_data['total_cells']
                }
                
                patients[patient_id]['images'].append(image_info)
                images[img_file.stem] = image_info
                
                # Update patient totals
                patients[patient_id]['total_parasitized'] += cell_counts['Parasitized']
                patients[patient_id]['total_uninfected'] += cell_counts['Uninfected']
                patients[patient_id]['total_wbc'] += cell_counts['White_Blood_Cell']
                patients[patient_id]['total_other'] += cell_counts['Other']
                
                # Update global totals
                total_stats['total_parasitized'] += cell_counts['Parasitized']
                total_stats['total_uninfected'] += cell_counts['Uninfected']
                total_stats['total_wbc'] += cell_counts['White_Blood_Cell']
                total_stats['total_other'] += cell_counts['Other']
    
    # Calculate final statistics
    total_stats['total_patients'] = len(patients)
    total_stats['total_images'] = len(images)
    
    print(f"\n✅ Indexing complete!")
    print(f"📊 Total patients: {total_stats['total_patients']}")
    print(f"📊 Total images: {total_stats['total_images']}")
    print(f"📊 Total cells: {sum([total_stats[k] for k in total_stats if k.startswith('total_') and k not in ['total_patients', 'total_images']])}")
    
    return patients, images, total_stats

def create_patient_splits(patients, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Create train/val/test splits ensuring no patient overlap"""
    
    print(f"\n📊 Creating patient splits...")
    print(f"📊 Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    random.seed(random_seed)
    patient_ids = list(patients.keys())
    random.shuffle(patient_ids)
    
    # Calculate split indices
    total_patients = len(patient_ids)
    train_end = int(total_patients * train_ratio)
    val_end = train_end + int(total_patients * val_ratio)
    
    train_patients = patient_ids[:train_end]
    val_patients = patient_ids[train_end:val_end]
    test_patients = patient_ids[val_end:]
    
    # Create split dictionaries with statistics
    splits = {}
    
    for split_name, patient_list in [('train', train_patients), ('val', val_patients), ('test', test_patients)]:
        split_stats = {
            'patients': len(patient_list),
            'images': 0,
            'parasitized_cells': 0,
            'uninfected_cells': 0,
            'wbc_cells': 0,
            'other_cells': 0
        }
        
        for patient_id in patient_list:
            patient_data = patients[patient_id]
            split_stats['images'] += len(patient_data['images'])
            split_stats['parasitized_cells'] += patient_data['total_parasitized']
            split_stats['uninfected_cells'] += patient_data['total_uninfected']
            split_stats['wbc_cells'] += patient_data['total_wbc']
            split_stats['other_cells'] += patient_data['total_other']
        
        splits[split_name] = {
            'patients': patient_list,
            'statistics': split_stats
        }
        
        print(f"📋 {split_name.upper()} SET:")
        print(f"   Patients: {split_stats['patients']}")
        print(f"   Images: {split_stats['images']}")
        print(f"   Parasitized cells: {split_stats['parasitized_cells']}")
        print(f"   Uninfected cells: {split_stats['uninfected_cells']}")
    
    return splits

def convert_to_yolo_format(annotation_data, img_width, img_height):
    """Convert annotations to YOLO format"""
    
    yolo_annotations = []
    
    # Class mapping
    class_map = {
        'Parasitized': 0,
        'Uninfected': 1,
        'White_Blood_Cell': 2
    }
    
    for cell in annotation_data['cells']:
        cell_type = cell['cell_type']
        coordinates = cell.get('coordinates', [])
        
        if cell_type not in class_map or not coordinates:
            continue
        
        class_id = class_map[cell_type]
        
        if cell['annotation_type'] == 'Polygon' and len(coordinates) >= 3:
            # Convert polygon to bounding box
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Normalize coordinates to [0, 1]
            center_x = (min_x + max_x) / 2 / img_width
            center_y = (min_y + max_y) / 2 / img_height
            width = (max_x - min_x) / img_width
            height = (max_y - min_y) / img_height
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        elif cell['annotation_type'] == 'Point' and len(coordinates) == 1:
            # Create small bounding box around point
            x, y = coordinates[0]
            box_size = 20  # pixels
            
            center_x = x / img_width
            center_y = y / img_height
            width = box_size / img_width
            height = box_size / img_height
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations

def copy_and_convert_images(raw_uncropped_path, processed_path, patients, splits):
    """Copy images and create YOLO annotations for each split"""
    
    print(f"\n📂 Copying images and creating YOLO annotations...")
    
    for split_name, split_data in splits.items():
        print(f"\n📋 Processing {split_name.upper()} split...")
        
        split_patients = split_data['patients']
        
        for patient_id in tqdm(split_patients, desc=f"Copying {split_name}"):
            patient_data = patients[patient_id]
            
            for image_info in patient_data['images']:
                # Source paths
                src_img_path = raw_uncropped_path / image_info['image_path']
                src_annotation_path = raw_uncropped_path / image_info['annotation_path']
                
                # Destination paths
                dst_img_path = processed_path / "yolo_annotations" / "images" / split_name / image_info['filename']
                dst_label_path = processed_path / "yolo_annotations" / "labels" / split_name / (Path(image_info['filename']).stem + '.txt')
                
                # Copy image
                if src_img_path.exists():
                    shutil.copy2(src_img_path, dst_img_path)
                
                # Convert and save annotations
                if src_annotation_path.exists():
                    annotation_data = parse_annotation_file(src_annotation_path)
                    if annotation_data:
                        yolo_annotations = convert_to_yolo_format(
                            annotation_data, 
                            image_info['image_width'], 
                            image_info['image_height']
                        )
                        
                        with open(dst_label_path, 'w') as f:
                            f.write('\n'.join(yolo_annotations))

def create_yolo_config(processed_path):
    """Create YOLO dataset configuration file"""
    
    config = {
        'path': str(processed_path / "yolo_annotations"),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 3,
        'names': ['parasitized', 'uninfected', 'white_blood_cell']
    }
    
    config_path = processed_path / "yolo_annotations" / "dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ YOLO config saved to: {config_path}")

def save_dataset_files(processed_path, patients, images, splits, total_stats):
    """Save dataset index and split information"""
    
    # Dataset index
    dataset_index = {
        'dataset_info': {
            'created_by': 'Shawn Jurgen Mayol',
            'date_created': 'July 2025',
            'dataset_path': 'data/raw/uncropped',
            'annotation_format': 'Both Polygon Set and Point Set',
            'description': 'Complete dataset using all available patient data'
        },
        'patients': patients,
        'images': images,
        'statistics': total_stats
    }
    
    index_path = processed_path / "dataset_index.json"
    with open(index_path, 'w') as f:
        json.dump(dataset_index, f, indent=2)
    
    # Dataset splits
    dataset_splits = {
        'metadata': {
            'split_ratios': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            },
            'random_seed': 42,
            'created_by': 'Shawn Jurgen Mayol',
            'date_created': 'July 2025',
            'total_patients': total_stats['total_patients'],
            'description': 'Patient-level splits ensuring no data leakage'
        }
    }
    dataset_splits.update(splits)
    
    splits_path = processed_path / "dataset_split.json"
    with open(splits_path, 'w') as f:
        json.dump(dataset_splits, f, indent=2)
    
    print(f"✅ Dataset index saved to: {index_path}")
    print(f"✅ Dataset splits saved to: {splits_path}")

def main():
    """Main data organization pipeline"""
    
    print("=" * 60)
    print("    COMPREHENSIVE MALARIA DATASET ORGANIZATION")
    print("    Using Both Polygon Set and Point Set Data")
    print("=" * 60)
    
    # Paths
    base_path = Path("data")
    raw_uncropped_path = base_path / "raw" / "uncropped"
    processed_path = base_path / "processed"
    
    # Verify source data exists
    if not raw_uncropped_path.exists():
        print(f"❌ Source data not found at {raw_uncropped_path}")
        return
    
    print(f"📂 Source data path: {raw_uncropped_path}")
    print(f"📂 Processed data path: {processed_path}")
    
    # Step 1: Create directory structure
    print(f"\n🔧 Step 1: Creating directory structure...")
    create_directory_structure(processed_path)
    
    # Step 2: Index all patients
    print(f"\n🔧 Step 2: Indexing all patients...")
    patients, images, total_stats = index_all_patients(raw_uncropped_path)
    
    # Step 3: Create splits
    print(f"\n🔧 Step 3: Creating train/val/test splits...")
    splits = create_patient_splits(patients)
    
    # Step 4: Copy images and create YOLO annotations
    print(f"\n🔧 Step 4: Copying images and creating YOLO annotations...")
    copy_and_convert_images(raw_uncropped_path, processed_path, patients, splits)
    
    # Step 5: Create YOLO configuration
    print(f"\n🔧 Step 5: Creating YOLO configuration...")
    create_yolo_config(processed_path)
    
    # Step 6: Save dataset files
    print(f"\n🔧 Step 6: Saving dataset metadata...")
    save_dataset_files(processed_path, patients, images, splits, total_stats)
    
    print(f"\n✅ Data organization complete!")
    print(f"📊 Final Statistics:")
    print(f"   Total patients: {total_stats['total_patients']}")
    print(f"   Total images: {total_stats['total_images']}")
    print(f"   Total cells: {sum([total_stats[k] for k in total_stats if k.startswith('total_') and k not in ['total_patients', 'total_images']])}")
    print(f"   Train patients: {len(splits['train']['patients'])}")
    print(f"   Val patients: {len(splits['val']['patients'])}")
    print(f"   Test patients: {len(splits['test']['patients'])}")

if __name__ == "__main__":
    main()
