# Stage1 Model

Multi-image to text generation model using CLIP features and T5 with LoRA fine-tuning.

> 目前 attention 在最终测试表现略好，mean 在 val 上出现过更高峰值。下一步建议优先在数据和融合结构上下功夫（更丰富的参考句 + cross-attention / object tags），然后尝试更大的语言模型与适度放宽 LoRA 约束。

## Model history

- image2haiku_20251104_103620 (v3-attention. 与图片信息对齐不够)

- image2haiku_20251104_101431 (v3-mean. 与图片信息对齐不够)

- image2haiku_20251103_151644 (v2. test 和 eval 输入图片数量不对 - 已通过 dataset 修复)

- image2haiku_20251103_144245 (v1. 重复生成现象 - 已通过超参数修复)

## Summary

- 架构：3 张图片 → 先用 CLIP 提取 3×512 特征 → 聚合（mean / attention）→ 线性变换 → T5 encoder-decoder（t5-small，LoRA 微调）→ 生成句子。
- 关键超参（两个实验一致，除 aggregation）：T5=t5-small，LoRA(r=8, alpha=32, dropout=0.1)，batch=16，epochs=100，lr=5e-5，warmup=200，scheduler=cosine，eval_interval=3，primary_metric=CIDEr，max_len=128。
- 可复现的工程点：仅微调 LoRA（参数占比较小），保存 JSON summary + 每次 eval 的 samples 便于定性分析。
- 资源/时长（本次小规模实验）：单次训练约 15–16 分钟（mean: ~950s，attention: ~937s），模型总参数约 62M，LoRA 可训练参数约 1.6M（mean）–2.13M（attention）。

## Architecture

```
3 Images → CLIP Features(3×512) → Aggregation → Linear → T5 Encoder-Decoder → Caption
                                      ↓
                          Mean Pooling / Attention Pooling
```

- **Anti-Repetition**: `repetition_penalty=1.2`, `no_repeat_ngram_size=3`
- **LoRA Fine-tuning**: Only 2.6% parameters trainable (1.6M/62M)
- **Enhanced Logging**: Detailed logs + JSON summaries with training metrics
- **Flexible Aggregation**: Runtime switchable between mean/attention pooling
- **Robust Generation**: Nucleus sampling + length penalty for diverse outputs

## Performance

- mean pooling（best epoch=96）

  - best val CIDEr: 0.3069（epoch 96）
  - final test: CIDEr=0.1990, METEOR=0.2323, BLEU-4=0.0533

- attention pooling（best epoch=75）

  - best val CIDEr: 0.2897（epoch 75）
  - final test: CIDEr=0.2189, METEOR=0.2375, BLEU-4=0.0580

- 两种模型都能生成语法通顺、场景级别合理的短句（适合描述宏观场景或主要对象），但常见问题包括：

  1. 泛化/模板化：句子偏向通用模板（"a man is ..."、"a group of ..."），细节稀缺；
  2. 对象/场景错配：有时会把图像中动物/物体混淆（giraffe ↔ cow 等），或把不同图像的信息拼接成不完整/不精确的描述；
  3. 低 n-gram 与一致性：BLEU-4/CIDEr 仍较低，说明与参考描述的具体匹配度有限。

- 结论要点：attention 在最终测试上略优（BLEU-4 与 CIDEr 均稍高），但 mean 达到过更高的单次 val CIDEr 峰值；总体表现受限于模型容量（t5-small）、图像-文本对齐策略与数据多样性。

### sample

> 来自 epoch_95_val_samples.json（mean）与 epoch_74_val_samples.json（attention）
> 说明：每条给出 group_id / image_ids / reference / prediction / metrics（METEOR,CIDEr,BLEU-1..4）以及短评。

#### Mean 实验 — Good

1. group_id: 162

- image_ids: ["000000054868.jpg","000000041962.jpg","000000356174.jpg"]
- reference: "A crowd watches as a young female tennis player jumps to hit a tennis ball on the court."
- prediction: "A tennis player is holding a tennis ball on the court, while a woman sits in front of a tennis court nearby."
- metrics: METEOR=0.5208, CIDEr=0.0, BLEU-4=0.31467
- comment: 好。动作与主体命中，表达简洁。

2. group_id: 386

- image_ids: ["000000269051.jpg","000000259253.jpg","000000050117.jpg"]
- reference: "A herd of black and brown cows is grazing on the grass in a field, accompanied by a tree."
- prediction: "A group of sheep grazes in a pasture near a grassy field, while nearby, a flock of cows is grazing on the grass."
- metrics: METEOR=0.55717, CIDEr=0.0, BLEU-4=0.25251
- comment: 好。场景语义抓得好。物种表述略有差异。

3. group_id: 738

- image_ids: ["000000210031.jpg","000000184834.jpg","000000061492.jpg"]
- reference: "A woman with a walker is sitting on a park bench, while a cat relaxes on a towel nearby, and a man is taking a selfie in front of a mirror as he brushes his teeth."
- prediction: "A dog sits on a couch in front of a cat, while a man is sitting in a chair near a desk, and a woman is holding a towel."
- metrics: METEOR=0.45256, CIDEr=0.0, BLEU-4=0.14991
- comment: 较好。局部对象捕捉到了。标签有偏差。

4. group_id: 554

- image_ids: ["000000337844.jpg","000000215826.jpg","000000317715.jpg"]
- reference: "In a park setting, a small Lego bench is positioned in front of a wooden bench..."
- prediction: "A parked car is parked on a sidewalk, while a bicycle is parked nearby, and a red bicycle is positioned in front of a wall."
- metrics: METEOR=0.31980, CIDEr=0.0, BLEU-4=0.14581
- comment: 好。主体相关性较好。

#### Mean 实验 — Bad

1. group_id: 715

- image_ids: ["000000231460.jpg","000000149464.jpg","000000563986.jpg"]
- reference: "At an intersection in the city, a pole displays various street name signs alongside a red traffic signal..."
- prediction: "A man is riding a motorcycle on a street near a street, while nearby, a pedestrian is parked in a parking lot."
- metrics: METEOR=0.10949, CIDEr=0.0, BLEU-4=0.01371
- comment: 失败。未抓住“标志/信号”。

1. group_id: 430

- image_ids: ["000000253202.jpg","000000225567.jpg","000000335909.jpg"]
- reference: "Three people are skiing in a snow-covered area..."
- prediction: "A skier and snowboarder are enjoying a snowy day..."
- metrics: METEOR=0.09950, CIDEr=0.0, BLEU-4=0.01770
- comment: 失败。细节缺失。

1. group_id: 14

- image_ids: ["000000368510.jpg","000000449708.jpg","000000236596.jpg"]
- reference: "A cluttered and messy desk features a small laptop surrounded by magazines and a can of soda..."
- prediction: "A computer is positioned on a table, while a computer displays a large screen of paper in the background."
- metrics: METEOR=0.15487, CIDEr=0.0, BLEU-4=0.01793
- comment: 失败。过于泛化。

1. group_id: 801

- image_ids: ["000000011887.jpg","000000177090.jpg","000000577632.jpg"]
- reference: "A motorcycle with a trailer is parked nearby a foggy street featuring a stop sign and a one-way sign..."
- prediction: "In a dark blue area, a small town is surrounded by trees..."
- metrics: METEOR=0.15845, CIDEr=0.0, BLEU-4=0.02926
- comment: 失败。场景拼接错误，室内/室外混淆。

#### Attention 实验 — Good

1. group_id: 162

- image_ids: ["000000054868.jpg","000000041962.jpg","000000356174.jpg"]
- reference: "A crowd watches as a young female tennis player jumps to hit a tennis ball on the court."
- prediction: "A tennis player is holding a tennis ball on the court..."
- metrics: METEOR=0.45027, CIDEr=0.0, BLEU-4=0.25306
- comment: 好。动作与主体匹配。

1. group_id: 738

- image_ids: ["000000210031.jpg","000000184834.jpg","000000061492.jpg"]
- reference: "A woman with a walker is sitting on a park bench..."
- prediction: "A dog is sitting on a couch..."
- metrics: METEOR=0.26652, CIDEr=0.0, BLEU-4=0.13599
- comment: 较好。句子流畅，对象标签仍错。

1. group_id: 14

- image_ids: ["000000368510.jpg","000000449708.jpg","000000236596.jpg"]
- reference: "A cluttered and messy desk features a small laptop..."
- prediction: "A computer is positioned on a table..."
- metrics: METEOR=0.41127, CIDEr=0.0, BLEU-4=0.12514
- comment: 好。关键对象覆盖，表达更连贯但仍缺细节。

1. group_id: 322

- image_ids: ["000000435671.jpg","000000269436.jpg","000000315163.jpg"]
- reference: "A small bird perches on a tree branch near a bird feeder..."
- prediction: "A small bird is laying on a tree..."
- metrics: METEOR=0.41487, CIDEr=0.0, BLEU-4=0.12514
- comment: 好。物体与位置关系较准确，语言自然。

#### Attention 实验 — Bad

1. group_id: 793

- image_ids: ["000000230892.jpg","000000485830.jpg","000000512531.jpg"]
- reference: "A small boy holds a surfboard in front of the ocean..."
- prediction: "A girl stands in a grassy field near a tree..."
- metrics: METEOR=0.14151, CIDEr=0.0, BLEU-4=0.01150
- comment: 差。scene/subject 严重偏移（海边 → 草地）。

2. group_id: 679

- image_ids: ["000000162768.jpg","000000186399.jpg","000000257066.jpg"]
- reference: "A traffic light displays a green figure indicating it's safe to walk..."
- prediction: "A man stands next to a street sign..."
- metrics: METEOR=0.14235, CIDEr=0.0, BLEU-4=0.01301
- comment: 差。只保留少量信息，细节丢失。缺少行人/信号/鸟等要素。

1. group_id: 520

- image_ids: ["000000116171.jpg","000000410774.jpg","000000542081.jpg"]
- reference: "In a vintage airplane museum, visitors stroll beneath displays of WWII-era planes..."
- prediction: "A group of people is flying a plane on a mountain..."
- metrics: METEOR=0.15974, CIDEr=0.0, BLEU-4=0.01417
- comment: 差。明显 hallucination（不现实的动作/场景组合）。

1. group_id: 430

- image_ids: ["000000253202.jpg","000000225567.jpg","000000335909.jpg"]
- reference: "Three people are skiing in a snow-covered area..."
- prediction: "A woman is skiing on a snowy slope..."
- metrics: METEOR=0.19512, CIDEr=0.0, BLEU-4=0.01821
- comment: 差。只保留笼统动作，缺失参考中的焦点或姿态。

### Available Metrics

- **Loss/**: Training and validation loss curves
- **BLEU/**: BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
- **Caption_Metrics/**: CIDEr and METEOR scores
- **Final_Test/**: Final evaluation results on test set

### CSV Export

All metrics are also exported to `training_metrics.csv` for analysis with pandas/plotting tools.

## Future improvement

1. 数据层（高优先级）

- 提升标注质量：补充更多、更详细的参考句（更少模板化、更多物体/动作描述）；针对长/复杂场景增加多参考句。
- 数据增强：文本 paraphrase、synonym 替换、CLIP 特征扰动（feature dropout）以增强鲁棒性。
- 加入视觉检测信号：用目标检测/标签（object tags）作为额外条件，减少物体混淆。

2. 模型与融合（中高优先级）

- 更强的语言端：试 t5-base / t5-large，会显著改善生成流畅度和细节表达（成本与显存上升）。
- 更细粒度的跨模态融合：将 CLIP 特征与 T5 做 cross-attention（而非先聚合成单一向量），或为每张图片加入位置/顺序 embedding。
- 更复杂的聚合器：从简单 mean/attention 升级为可学习的加权聚合或层次化 attention（image-level → patch/region-level）。
- 考虑对 CLIP encoder 做微调或使用更强的视觉 encoder（ViT 大小、或 region-based 特征）。

3. 训练与 LoRA 策略（中优先级）

- 调整 LoRA 配置（r 增大或分层插入）以提升表达力；尝试不同的冻结/解冻策略。
- 增大 batch / 更长训练（当前只在有限条件下跑到 100 epochs），并对 lr schedule、warmup 进行小范围搜索。
- 如果可行，使用对比损失（image-text contrastive）与生成损失联合训练，强化图文语义对齐。

4. 解码与后处理（低—中优先级）

- 更稳健的解码策略：微调 nucleus sampling / top-k / diverse beam，会影响多样性与精确性权衡。
- 后处理：基于检测/实体一致性做简单过滤或重写，减少明显的物体错配。
