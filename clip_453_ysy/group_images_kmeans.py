import numpy as np
import json
from sklearn.cluster import KMeans
from tqdm import tqdm
import random

# === 路径设置 ===
features_path = "image_features.npy"
ids_path = "image_ids.json"
captions_path = "coco_subset/captions_subset.json"
output_path = "image_groups_with_captions.json"

# === Step 1. 加载数据 ===
print("🔹 加载视觉特征、图片ID与caption中...")
features = np.load(features_path)

with open(ids_path, "r") as f:
    img_ids = json.load(f)

with open(captions_path, "r") as f:
    captions_data = json.load(f)

# captions_subset.json 格式假设为：
# [
#   {"img_id": "000000123456.jpg", "captions": ["a cat on a mat", "a small kitten resting", ...]},
#   ...
# ]

# 构建一个 img_id → captions 的快速索引
caption_map = {item["img_id"]: item["captions"] for item in captions_data}

assert len(features) == len(img_ids), "❌ features 与 img_ids 数量不一致！"
n_images = len(img_ids)
n_clusters = n_images // 3  # 1000 组
print(f"✅ 共 {n_images} 张图片，将分为 {n_clusters} 组，每组三张。")

# === Step 2. KMeans 聚类 ===
print("🔹 正在进行 KMeans 聚类...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
kmeans.fit(features)
centers = kmeans.cluster_centers_

# === Step 3. 每个中心取3张最近的图片 ===
print("🔹 为每个聚类中心分配3张最相似图片...")
used_indices = set()
groups = []

for i in tqdm(range(n_clusters)):
    available_idx = [idx for idx in range(n_images) if idx not in used_indices]
    available_features = features[available_idx]
    distances = np.linalg.norm(available_features - centers[i], axis=1)

    # 取最近3张
    top3_indices = np.argsort(distances)[:3]
    selected_idx = [available_idx[j] for j in top3_indices]
    used_indices.update(selected_idx)

    group_images = []
    for j in selected_idx:
        img_id = img_ids[j]
        captions = caption_map.get(img_id, [])
        group_images.append({
            "img_id": img_id,
            "captions": captions
        })

    groups.append({
        "group_id": i,
        "images": group_images
    })

# === Step 4. 若有未分配图片，随机补齐到不足3的组 ===
remaining = [idx for idx in range(n_images) if idx not in used_indices]
if remaining:
    print(f"⚠️ 有 {len(remaining)} 张图片未分配，将随机补充入组。")
    random.shuffle(remaining)
    for g in groups:
        while len(g["images"]) < 3 and remaining:
            j = remaining.pop()
            img_id = img_ids[j]
            captions = caption_map.get(img_id, [])
            g["images"].append({
                "img_id": img_id,
                "captions": captions
            })

# === Step 5. 保存结果 ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(groups, f, indent=2, ensure_ascii=False)

print(f"✅ 已完成分组，共 {len(groups)} 组，每组3张图片。")
print(f"📁 输出文件：{output_path}")
