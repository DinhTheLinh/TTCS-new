import os
import json
import torch
import clip
from PIL import Image

model_path = "clip_triplet_hard_margin_0_4_2.pth"
photo_root = r"photo/photo/photo"

save_matrix = "photo_embeddings_hard_2.pt"
save_photo_path_json = "photo_paths_hard_2.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Loaded model:", model_path)


photo_paths = []

for class_name in os.listdir(photo_root):
    class_dir = os.path.join(photo_root, class_name)

    if not os.path.isdir(class_dir):
        continue

    for file in os.listdir(class_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            full_path = os.path.join(class_dir, file)
            photo_paths.append(full_path)

print("Total photos:", len(photo_paths))


all_features = []

with torch.no_grad():
    for idx, path in enumerate(photo_paths):
        image = Image.open(path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)

        feat = model.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        all_features.append(feat.cpu())

        if (idx + 1) % 100 == 0:
            print(f"Encoded {idx+1}/{len(photo_paths)}")

photo_features = torch.cat(all_features, dim=0)

print("Feature shape:", photo_features.shape)


torch.save(photo_features, save_matrix)

with open(save_photo_path_json, "w", encoding="utf-8") as f:
    json.dump(photo_paths, f, indent=2)

print("Saved:", save_matrix)
print("Saved:", save_photo_path_json)