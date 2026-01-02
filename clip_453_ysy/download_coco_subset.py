#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 COCO 2017 子集（图片 + Captions）
修正版：保证 captions_subset.json 与 images 完全对应
作者: 叶圣尧
"""

import os
import json
import random
import requests
from tqdm import tqdm

# ======================
# ⚙️ 参数设置
# ======================
NUM_IMAGES = 3000
OUT_DIR = "./coco_subset"
ANNOTATION_FILE = "captions_train2017.json"
ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
BASE_URL = "http://images.cocodataset.org/train2017/"

os.makedirs(f"{OUT_DIR}/images", exist_ok=True)

# ======================
# 📦 Step 1. 下载 caption 文件
# ======================
if not os.path.exists(ANNOTATION_FILE):
    print("⬇️ 正在下载 captions_train2017.json ...")
    os.system("wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    os.system("unzip -q annotations_trainval2017.zip 'annotations/captions_train2017.json'")
    os.system("mv annotations/captions_train2017.json .")
    os.system("rm -r annotations annotations_trainval2017.zip")

# ======================
# 📖 Step 2. 加载数据
# ======================
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

print(f"📸 总图片数: {len(images)}, 总caption数: {len(annotations)}")

# 建立映射：image_id -> file_name
id_to_name = {img["id"]: img["file_name"] for img in images}

# ======================
# 🎯 Step 3. 随机抽取3000张图片
# ======================
subset_images = random.sample(images, NUM_IMAGES)
subset_ids = {img["id"] for img in subset_images}
subset_fnames = {img["file_name"] for img in subset_images}

# ======================
# 💬 Step 4. 收集caption（安全映射）
# ======================
captions_dict = {name: [] for name in subset_fnames}
for ann in annotations:
    img_id = ann["image_id"]
    if img_id in subset_ids:
        fname = id_to_name[img_id]
        captions_dict[fname].append(ann["caption"])

# ======================
# 🖼 Step 5. 下载图片
# ======================
for fname in tqdm(subset_fnames, desc="Downloading images"):
    url = BASE_URL + fname
    save_path = os.path.join(OUT_DIR, "images", fname)
    if not os.path.exists(save_path):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"❌ 下载失败: {fname} (status {r.status_code})")
        except Exception as e:
            print(f"⚠️ 网络错误: {fname} ({e})")

# ======================
# 💾 Step 6. 保存 captions_subset.json
# ======================
subset_data = []
for fname in sorted(subset_fnames):
    subset_data.append({
        "img_id": fname,
        "captions": captions_dict[fname]
    })

out_json = os.path.join(OUT_DIR, "captions_subset.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(subset_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 共保存 {len(subset_data)} 张图片")
print(f"✅ captions_subset.json 位于: {out_json}")
print(f"✅ 图片保存路径: {OUT_DIR}/images/")
