python scripts/test_best_model.py \
    --experiment_dir outputs/image2haiku_20251106_125849 \
    --features_path data/image_features.npy \
    --image_ids_path data/image_ids.json \
    --annotations_path data/image_groups_with_1caption.json


---
# Best Model Testing Scripts

这些脚本允许你使用训练过程中保存的最佳 checkpoint 进行测试，而无需修改训练代码。

## 脚本说明

### 1. `test_best_model.py` - 单个实验测试

测试指定实验的最佳 checkpoint。

**使用方法：**

```bash
# 测试指定实验
python scripts/test_best_model.py \
    --experiment_dir outputs/image2haiku_20241105_123456 \
    --features_path data/image_features.npy \
    --image_ids_path data/image_ids.json \
    --annotations_path data/image_groups_with_1caption.json

# 自动使用最新实验
python scripts/test_best_model.py \
    --features_path data/image_features.npy \
    --image_ids_path data/image_ids.json \
    --annotations_path data/image_groups_with_1caption.json

# 限制测试样本数量（用于快速测试）
python scripts/test_best_model.py \
    --experiment_dir outputs/image2haiku_20241105_123456 \
    --features_path data/image_features.npy \
    --image_ids_path data/image_ids.json \
    --annotations_path data/image_groups_with_1caption.json \
    --max_samples 100
```

### 2. `batch_test_experiments.py` - 批量测试

测试 outputs 目录下所有实验的最佳 checkpoint。

**使用方法：**

```bash
# 批量测试所有实验
python scripts/batch_test_experiments.py \
    --data_config data_config.json

# 指定outputs目录
python scripts/batch_test_experiments.py \
    --outputs_dir my_experiments \
    --data_config data_config.json

# 限制每个实验的测试样本数
python scripts/batch_test_experiments.py \
    --data_config data_config.json \
    --max_samples 200
```

**数据配置文件格式** (`data_config.json`):

```json
{
  "features_path": "data/image_features.npy",
  "image_ids_path": "data/image_ids.json",
  "annotations_path": "data/image_groups_with_1caption.json"
}
```

## 输出结果

### 单个实验测试输出

- `experiment_dir/test_results/best_model_test_results_TIMESTAMP.json` - 测试指标和配置信息
- `experiment_dir/test_results/best_model_test_samples_TIMESTAMP.json` - 生成样本详情

### 批量测试输出

- `outputs/batch_test_summary.json` - 所有实验的汇总对比
- 每个实验目录下的单独测试结果

## 关键特性

✅ **使用最佳 checkpoint**: 自动加载验证集上表现最好的模型权重  
✅ **保留训练配置**: 从 checkpoint 恢复完整的模型配置  
✅ **详细结果记录**: 保存完整的测试指标、样本和元数据  
✅ **批量对比**: 支持多个实验的性能对比  
✅ **错误处理**: 优雅处理缺失文件或配置错误

## 与训练流程的区别

| 方面     | 训练中的 test         | 独立测试脚本             |
| -------- | --------------------- | ------------------------ |
| 使用模型 | 最后一个 epoch 的模型 | 最佳 checkpoint 模型     |
| 测试时机 | 训练结束后立即执行    | 任意时间独立执行         |
| 结果保存 | 混合在训练日志中      | 独立的 test_results 目录 |
| 灵活性   | 固定流程              | 可配置样本数、多实验对比 |

## 使用建议

1. **单次训练后**: 使用 `test_best_model.py` 获取最佳模型的真实测试性能
2. **多次实验对比**: 使用 `batch_test_experiments.py` 进行横向对比
3. **快速验证**: 设置 `--max_samples` 参数进行小规模测试
4. **结果分析**: 查看 JSON 文件进行详细的性能分析和可视化
