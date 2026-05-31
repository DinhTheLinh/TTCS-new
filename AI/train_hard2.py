import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip

from triplet_dataset_hard import HardTripletDataset

def calculate_val_map(all_anchors, all_positives, all_labels):
    similarity_matrix = all_anchors @ all_positives.T
    num_queries = similarity_matrix.size(0)
    ap_sum = 0.0
    
    for i in range(num_queries):
        true_label = all_labels[i]
        sim_scores = similarity_matrix[i]
        sorted_indices = torch.argsort(sim_scores, descending=True)
        
        retrieved_labels = [all_labels[idx] for idx in sorted_indices]
        total_relevant = retrieved_labels.count(true_label)
        
        hits = 0
        sum_precisions = 0.0
        for rank, label in enumerate(retrieved_labels):
            if label == true_label:
                hits += 1
                sum_precisions += hits / (rank + 1.0)
        
        if total_relevant > 0:
            ap_sum += sum_precisions / total_relevant
            
    return (ap_sum / num_queries) * 100

parser = argparse.ArgumentParser()
parser.add_argument("--train_json", type=str, default="train_triplets_hard_train.json")
parser.add_argument("--val_json", type=str, default="train_triplets_hard_val.json")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--margin", type=float, default=0.4)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--save_path", type=str, default="clip_triplet_hard_margin_0_4.pth")
parser.add_argument("--baseline_model_path", type=str, default="clip_triplet_margin_0_4.pth")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()
try:
    model.load_state_dict(torch.load(args.baseline_model_path, map_location=device))
except:
    print("Không tìm thấy file baseline model")

train_dataset = HardTripletDataset(args.train_json, preprocess, is_train=True)
val_dataset = HardTripletDataset(args.val_json, preprocess, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_val_loss = float("inf")
no_improve_count = 0

for epoch in range(args.max_epochs):
    model.train()
    train_loss = 0.0

    for batch_idx, (anchor, positive, offline_negative, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train")):
        anchor, positive, offline_negative = anchor.to(device), positive.to(device), offline_negative.to(device)

        anchor_feat = model.encode_image(anchor)
        positive_feat = model.encode_image(positive)
        offline_neg_feat = model.encode_image(offline_negative)
        
        anchor_feat = anchor_feat / anchor_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        positive_feat = positive_feat / positive_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        offline_neg_feat = offline_neg_feat / offline_neg_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        sim_matrix = anchor_feat @ positive_feat.T
        
        hard_negatives = []
        for i in range(len(labels)):
            mask = [l != labels[i] for l in labels]
            mask = torch.tensor(mask).to(device)
            
            if mask.sum() == 0:
                hard_negatives.append(offline_neg_feat[i])
            else:
                masked_sim = sim_matrix[i].clone()
                masked_sim[~mask] = -1.0
                hardest_idx = masked_sim.argmax()
                hard_negatives.append(positive_feat[hardest_idx])
                
        online_hard_neg_feat = torch.stack(hard_negatives)

        loss = criterion(anchor_feat, positive_feat, online_hard_neg_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_anchor_features = []
    val_positive_features = []
    val_labels_list = []

    with torch.no_grad():
        for anchor, positive, offline_negative, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
            anchor, positive, offline_negative = anchor.to(device), positive.to(device), offline_negative.to(device)

            anchor_feat = model.encode_image(anchor)
            positive_feat = model.encode_image(positive)
            offline_neg_feat = model.encode_image(offline_negative)

            anchor_feat = anchor_feat / anchor_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            positive_feat = positive_feat / positive_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            offline_neg_feat = offline_neg_feat / offline_neg_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            loss = criterion(anchor_feat, positive_feat, offline_neg_feat)
            val_loss += loss.item()

            val_anchor_features.append(anchor_feat.cpu())
            val_positive_features.append(positive_feat.cpu())
            val_labels_list.extend(labels)

    avg_val_loss = val_loss / len(val_loader)

    all_anchors = torch.cat(val_anchor_features, dim=0)
    all_positives = torch.cat(val_positive_features, dim=0)
    
    epoch_val_map = calculate_val_map(all_anchors, all_positives, val_labels_list)

    print(f"Epoch [{epoch + 1}/{args.max_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mAP: {epoch_val_map:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_count = 0
        torch.save(model.state_dict(), args.save_path)
        print(f"Đã lưu model vào: {args.save_path}")
    else:
        no_improve_count += 1
        print(f"Val loss không cải thiện. ({no_improve_count}/{args.patience})")

    if no_improve_count >= args.patience:
        print("Dừng train vì AI đã đạt giới hạn!")
        break