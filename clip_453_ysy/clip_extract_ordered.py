import torch
import clip
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import os

# 路径配置
image_dir = "coco_subset/images"
caption_json = "coco_subset/captions_subset.json"
features_path = "image_features.npy"
ids_path = "image_ids.json"

# 1. 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. 读取 captions_subset.json 中的顺序
with open(caption_json, 'r') as f:
    data = json.load(f)

# 获取按顺序排列的 img_id
img_ids = [item["img_id"] for item in data]

# 3. 提取图像特征
features = []
valid_ids = []

for img_id in tqdm(img_ids, desc="Extracting image features"):
    img_path = os.path.join(image_dir, img_id)
    if not os.path.exists(img_path):
        print(f"⚠️ 图片不存在: {img_id}")
        continue
    try:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat /= feat.norm(dim=-1, keepdim=True)
        features.append(feat.cpu().numpy())
        valid_ids.append(img_id)
    except Exception as e:
        print(f"⚠️ 无法处理 {img_id}: {e}")

# 4. 保存结果
features = np.concatenate(features, axis=0)
np.save(features_path, features)

with open(ids_path, 'w') as f:
    json.dump(valid_ids, f, indent=2)

print(f"✅ 已保存 {len(valid_ids)} 张图片的特征到 {features_path}")
print(f"✅ 对应的 img_id 已保存到 {ids_path}")
