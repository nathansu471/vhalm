import json
from collections import defaultdict
from tqdm import tqdm

# === 1. 加载原始 COCO captions 文件 ===
input_file = "captions_train2017.json"
output_file = "captions_summary.json"

print(f"📂 正在读取 {input_file} ...")
with open(input_file, "r") as f:
    data = json.load(f)

# === 2. 建立 image_id → file_name 映射 ===
id_to_name = {img["id"]: img["file_name"] for img in data["images"]}

# === 3. 按 image_id 聚合 captions ===
captions_dict = defaultdict(list)
for ann in tqdm(data["annotations"], desc="整理 captions"):
    img_id = ann["image_id"]
    cap = ann["caption"].strip()
    captions_dict[img_id].append(cap)

# === 4. 构造输出数据结构 ===
output_data = []
for img in data["images"]:
    img_id = img["id"]
    caps = captions_dict.get(img_id, [])
    if caps:
        output_data.append({
            "image_id": img_id,
            "file_name": img["file_name"],
            "captions": caps[:5]
        })


# === 5. 保存为 JSON 文件 ===
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 已保存到 {output_file}")
print(f"共包含 {len(output_data)} 张图片。")
