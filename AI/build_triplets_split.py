import os
import json
import random
import torch
import clip
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device đang chạy:", device)

model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float().eval()

baseline_path = "clip_triplet_margin_0_4.pth"
if os.path.exists(baseline_path):
    model.load_state_dict(torch.load(baseline_path, map_location=device))
    print(f"Đã load baseline model: {baseline_path}")
else:
    print("Không tìm thấy baseline model")

def encode_image(image_path):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feat.squeeze(0).cpu()
    except Exception as e:
        print(f"Lỗi khi đọc file {image_path}: {e}")
        return None

with open("dataset_split.json", "r", encoding="utf-8") as f:
    splits = json.load(f)

def build_triplet_json(split_name, output_filename):
    
    sketch_data = splits[split_name]["sketch"]
    photo_data = splits[split_name]["photo"]
    
    valid_classes = sorted([cls for cls in sketch_data.keys() if len(sketch_data[cls]) > 0 and len(photo_data[cls]) > 0])
    
    photo_features = {}
    all_photo_items = []
    for cls in tqdm(valid_classes):
        for path in photo_data[cls]:
            feat = encode_image(path)
            if feat is not None:
                photo_features[path] = feat
                all_photo_items.append((path, cls))
                
    triplets = []
    for cls in tqdm(valid_classes):
        for sketch_path in sketch_data[cls]:
            anchor_feat = encode_image(sketch_path)
            if anchor_feat is None: continue
            
            positive_path = random.choice(photo_data[cls])
            
            best_neg_path, best_neg_class, best_sim = None, None, -1e9
            
            for photo_path, photo_cls in all_photo_items:
                if photo_cls == cls: continue 
                
                sim = torch.matmul(anchor_feat, photo_features[photo_path]).item()
                if sim > best_sim:
                    best_sim = sim
                    best_neg_path = photo_path
                    best_neg_class = photo_cls
                    
            triplets.append({
                "anchor": sketch_path,
                "positive": positive_path,
                "negative": best_neg_path,
                "class_name": cls,
                "negative_class": best_neg_class,
                "negative_similarity": best_sim
            })
            
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
        
build_triplet_json("train", "train_triplets_hard_train.json")
build_triplet_json("val", "train_triplets_hard_val.json")