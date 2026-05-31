import torch
import clip
import json
from PIL import Image
import os
from tqdm import tqdm

def main():
    model_path = "clip_triplet_hard_margin_0_4.pth"  
    split_json_path = "dataset_split.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy trên: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.float()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print("Không tìm thấy model đã train, đang dùng model gốc")
    model.eval()

    with open(split_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)


    photo_paths = []
    photo_labels = []
    
    for split_name in ["train", "val", "test"]:
        for cls_name, paths in splits[split_name]["photo"].items():
            for path in paths:
                photo_paths.append(path)
                photo_labels.append(cls_name)
                
    print(f"Tổng số ảnh trong kho tìm kiếm: {len(photo_paths)}")
    photo_features_list = []
    with torch.no_grad():
        for path in tqdm(photo_paths, desc="Encoding Gallery"):
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            photo_features_list.append(feat.cpu())
            
    photo_features = torch.cat(photo_features_list, dim=0)

    sketch_paths = []
    sketch_labels = []
    
    for cls_name, paths in splits["test"]["sketch"].items():
        for path in paths:
            sketch_paths.append(path)
            sketch_labels.append(cls_name)

    print(f"Tổng số ảnh phác thảo dùng để thi: {len(sketch_paths)}")

    top1, top5, top10 = 0, 0, 0
    ap_sum = 0.0
    with torch.no_grad():
        for i, sketch_path in enumerate(tqdm(sketch_paths, desc="Chấm Test mAP")):
            image = preprocess(Image.open(sketch_path).convert("RGB")).unsqueeze(0).to(device)

            sketch_feat = model.encode_image(image)
            sketch_feat = sketch_feat / sketch_feat.norm(dim=-1, keepdim=True)

            similarity = (sketch_feat.cpu() @ photo_features.T).squeeze()
            sorted_indices = similarity.argsort(descending=True)
            
            true_label = sketch_labels[i]
            retrieved_labels = [photo_labels[idx] for idx in sorted_indices]

            if true_label in retrieved_labels[:1]: top1 += 1
            if true_label in retrieved_labels[:5]: top5 += 1
            if true_label in retrieved_labels[:10]: top10 += 1

            hits = 0
            sum_precisions = 0.0
            total_relevant = photo_labels.count(true_label)

            for rank, label in enumerate(retrieved_labels):
                if label == true_label:
                    hits += 1
                    sum_precisions += hits / (rank + 1.0)

            if total_relevant > 0:
                ap_sum += sum_precisions / total_relevant

    total_queries = len(sketch_paths)
    print(f"Độ chính xác Top-1:  {(top1 / total_queries) * 100:.2f}%")
    print(f"Độ chính xác Top-5:  {(top5 / total_queries) * 100:.2f}%")
    print(f"Độ chính xác Top-10: {(top10 / total_queries) * 100:.2f}%")
    print(f"mAP: {(ap_sum / total_queries) * 100:.2f}%")

if __name__ == "__main__":
    main()