import asyncio
from openai import AsyncOpenAI
import json, random, os
from tqdm.asyncio import tqdm_asyncio

# ======================
# ⚙️ 配置部分
# ======================
API_KEY = "your_openai_api_key_here"  # 替换为你的 OpenAI API Key
MODEL_NAME = "gpt-4o-mini"     # 推荐模型
CONCURRENCY = 10               # 并发数量
SAVE_INTERVAL = 50             # 每多少条保存一次
TMP_FILE = "merged_captions_tmp.json"
OUT_FILE = "merged_captions_async.json"

client = AsyncOpenAI(api_key=API_KEY)

# ======================
# 📂 载入数据与断点续跑
# ======================
with open("data/image_groups_with_captions.json", "r", encoding="utf-8") as f:
    groups = json.load(f)

results = []
start_idx = 0

if os.path.exists(TMP_FILE):
    with open(TMP_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    start_idx = len(results)
    print(f"✅ 检测到已有 {start_idx} 条结果，将从第 {start_idx+1} 组继续。")

groups = groups[start_idx:]

# ======================
# 🧠 定义生成函数
# ======================
async def merge_one_group(group, idx):
    captions = [random.choice(img["captions"]) for img in group["images"]]

    prompt = f"""
    You are a vision-language expert.
    You will be given three image captions describing similar or related scenes.
    Please merge them into ONE coherent and concise English description,
    avoiding repetition but preserving all important details.

    Captions:
    1. {captions[0]}
    2. {captions[1]}
    3. {captions[2]}

    Output:
    """

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=60,
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

