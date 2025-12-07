import asyncio
from openai import AsyncOpenAI
import json, random, os
from tqdm.asyncio import tqdm_asyncio

# ======================
# ⚙️ 配置部分
# ======================
API_KEY = "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"  # 替换为你的 OpenAI API Key
MODEL_NAME = "gpt-4o"     # 推荐模型
CONCURRENCY = 10               # 并发数量
SAVE_INTERVAL = 50             # 每多少条保存一次
TMP_FILE = "merged_captions_tmp.json"
OUT_FILE = "merged_captions_async3.json"

client = AsyncOpenAI(api_key=API_KEY)

# ======================
# 📂 载入数据与断点续跑
# ======================
with open("data/image_groups_curriculum.json", "r", encoding="utf-8") as f:
    groups = json.load(f)

results = []
start_idx = 0

# if os.path.exists(TMP_FILE):
#     with open(TMP_FILE, "r", encoding="utf-8") as f:
#         results = json.load(f)
#     start_idx = len(results)
#     print(f"✅ 检测到已有 {start_idx} 条结果，将从第 {start_idx+1} 组继续。")

groups = groups[start_idx:]

# ======================
# 🧠 定义生成函数
# ======================
async def merge_one_group(group, idx):
    captions = [random.choice(img["captions"]) for img in group["images"]]

    prompt = f"""
    You are a vision-language expert. Your task is to synthesize three image captions into ONE unified, natural English description.
    Important rules:
    - DO NOT simply concatenate captions with "and" or commas
    - Identify the COMMON theme, subject, or scene across all three captions
    - Merge overlapping information into a single coherent statement
    - Highlight unique details that appear in only one or two captions
    - Write as if describing a single unified scene, not three separate images
    
    Example 1:
    Input captions:
    1. A red car parked on the street
    2. A red vehicle next to a building
    3. A car with red paint in an urban setting

    Bad output (simple concatenation): "A red car parked on the street and a red vehicle next to a building and a car with red paint in an urban setting"
    Good output: "A red car parked on an urban street next to a building"

    Example 2:
    Input captions:
    1. A dog playing in the park
    2. A golden retriever running on grass
    3. A dog fetching a ball outdoors

    Bad output: "A dog playing in the park and a golden retriever running on grass and a dog fetching a ball outdoors"
    Good output: "A golden retriever playing fetch with a ball in a grassy park"

    Example 3:
    Input captions:
    1. Sunset over mountains
    2. Orange sky above peaks
    3. Evening light on mountain range

    Bad output: "Sunset over mountains and orange sky above peaks and evening light on mountain range"
    Good output: "A mountain range at sunset with an orange sky and evening light"
    

    Captions:
    1. {captions[0]}
    2. {captions[1]}
    3. {captions[2]}

    Output a single, natural English sentence that captures the essence of all three captions:
    """

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=120,
        )
        merged = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error idx {idx}] {e}")
        merged = ""

    result = {
        "group_id": group["group_id"],
        "image_ids": [img["img_id"] for img in group["images"]],
        "captions": captions,
        "merged_caption": merged
    }
    return result

# ======================
# 🧵 异步并发任务调度
# ======================
async def main():
    sem = asyncio.Semaphore(CONCURRENCY)

    async def sem_task(idx, group):
        async with sem:
            res = await merge_one_group(group, idx)
            results.append(res)
            if len(results) % SAVE_INTERVAL == 0:
                with open(TMP_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            return res

    await tqdm_asyncio.gather(*[
        sem_task(i + start_idx, group)
        for i, group in enumerate(groups)
    ])

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 完成！共生成 {len(results)} 条融合描述。")

# ======================
# 🚀 启动
# ======================
if __name__ == "__main__":
    asyncio.run(main())
