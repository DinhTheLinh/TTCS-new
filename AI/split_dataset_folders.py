import os
import json
import random
from collections import defaultdict

sketch_root = "./sketch/sketch"
photo_root = "./photo/photo"
random.seed(42)

def split_domain_data(root_dir):
    train_files, val_files, test_files = defaultdict(list), defaultdict(list), defaultdict(list)
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    for class_name in classes:
        class_path = os.path.join(root_dir, class_name)
        files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        random.shuffle(files)
        
        n = len(files)
        train_end = int(0.7 * n)
        val_end = train_end + int(0.15 * n)
        
        train_files[class_name] = files[:train_end]
        val_files[class_name] = files[train_end:val_end]
        test_files[class_name] = files[val_end:]
        
    return train_files, val_files, test_files

sketch_train, sketch_val, sketch_test = split_domain_data(sketch_root)
photo_train, photo_val, photo_test = split_domain_data(photo_root)

dataset_split = {
    "train": {"sketch": sketch_train, "photo": photo_train},
    "val": {"sketch": sketch_val, "photo": photo_val},
    "test": {"sketch": sketch_test, "photo": photo_test}
}

with open("dataset_split.json", "w", encoding="utf-8") as f:
    json.dump(dataset_split, f, ensure_ascii=False, indent=2)