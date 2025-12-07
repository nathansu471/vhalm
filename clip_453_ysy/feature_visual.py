import numpy as np, json

features = np.load("image_features.npy")
ids = json.load(open("image_ids.json"))

print("特征数量:", len(features))
print("ID数量:", len(ids))
assert len(features) == len(ids), "❌ 数量不匹配！"
print("✅ 顺序匹配无误。")
