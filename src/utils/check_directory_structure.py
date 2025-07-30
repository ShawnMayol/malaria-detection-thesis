"""
Directory Structure Checker
Helper script to identify the actual structure of the uncropped dataset
"""

import os
from pathlib import Path

def check_directory_structure(base_path="data/raw/uncropped"):
    """Check and display the directory structure"""
    path = Path(base_path)
    
    print(f"🔍 Checking directory structure at: {path.absolute()}")
    print(f"📁 Directory exists: {path.exists()}")
    
    if not path.exists():
        print("❌ Directory not found!")
        return None
    
    print("\n📂 Directory contents:")
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        """Print directory tree structure"""
        if current_depth >= max_depth:
            return
            
        try:
            items = sorted(directory.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                
                print(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and current_depth < max_depth - 1:
                    extension = "    " if is_last else "│   "
                    print_tree(item, prefix + extension, max_depth, current_depth + 1)
                    
        except PermissionError:
            print(f"{prefix}└── [Permission Denied]")
    
    print_tree(path)
    
    # Look for NIH dataset specifically
    print("\n🔍 Looking for NIH dataset patterns...")
    
    # Check common variations
    possible_paths = [
        path / "NIH-NLM-ThinBloodSmearsPf",
        path / "NIH-NLM-ThinBloodSmearsPf-main", 
        path / "malaria",
        path / "dataset",
        path / "uncropped"
    ]
    
    for possible_path in possible_paths:
        if possible_path.exists():
            print(f"✅ Found potential dataset at: {possible_path}")
            
            # Check for Polygon Set and Point Set
            polygon_path = possible_path / "Polygon Set"
            point_path = possible_path / "Point Set"
            
            if polygon_path.exists():
                print(f"   📁 Polygon Set found: {polygon_path}")
                
                # Count patients in polygon set
                patient_dirs = [d for d in polygon_path.iterdir() if d.is_dir()]
                print(f"   👥 Patients in Polygon Set: {len(patient_dirs)}")
                
                # Check structure of first patient
                if patient_dirs:
                    first_patient = patient_dirs[0]
                    img_dir = first_patient / "img"
                    gt_dir = first_patient / "GT"
                    
                    print(f"   📁 Sample patient structure ({first_patient.name}):")
                    print(f"      📁 img/ exists: {img_dir.exists()}")
                    print(f"      📁 GT/ exists: {gt_dir.exists()}")
                    
                    if img_dir.exists():
                        images = list(img_dir.glob("*.jpg"))
                        print(f"      🖼️  Images: {len(images)}")
                    
                    if gt_dir.exists():
                        annotations = list(gt_dir.glob("*.txt"))
                        print(f"      📝 Annotations: {len(annotations)}")
            
            if point_path.exists():
                print(f"   📁 Point Set found: {point_path}")
    
    # Also check if files are directly in uncropped
    direct_files = [f for f in path.iterdir() if f.is_file()]
    if direct_files:
        print(f"\n📄 Direct files in uncropped/: {len(direct_files)}")
        for f in direct_files[:5]:  # Show first 5 files
            print(f"   - {f.name}")
    
    return path

if __name__ == "__main__":
    check_directory_structure()