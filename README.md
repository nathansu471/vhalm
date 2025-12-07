# VHaLM: From Visual Moments to Haiku

# Visual-Haiku Language Model for Multi-Image Poetic Generation

## Image Extraction & Captions

### Step Overview

| Stage               | Input Data                | Output Data                                            | Description                                  |
| ------------------- | ------------------------- | ------------------------------------------------------ | -------------------------------------------- |
| Image Download      | COCO official URLs        | `coco_subset/images/*.jpg`                             | Ensure consistency between images and JSON   |
| Caption Structuring | Original multi-field JSON | `captions_subset.json`: multiple captions per `img_id` | Simplify to a clean pair format              |
| Feature Extraction  | Image pixels              | `image_features.npy` + `image_ids.json`                | CLIP visual encoder outputs                  |
| Cluster Grouping    | Image feature matrix      | `image_groups_with_captions.json`                      | 1,000 groups × 3 semantically similar images |

### File dir structure (up-to-date)

```
./
  data/
    test_data/ 			# Test data for midway code
    	model1/			# For stage 1 model(finetuning)
            annotations/
            features/
            images/
  models/				# Paras of trained models
  src/					# Source code
    model1/				# For stage 1 model(finetuning)
        outputs/
            checkpoints/
            logs/
            samples/
        dataset.py
        eval.py
        model.py
        train.py
  results/				# For model eval results (data & img)
```

### Prompt

> output 1 description for 3 images (model 1 test data)

```
Below are three image about similar scenes. Write ONE concise(< 50 tokens), objective sentence that summarizes what all three images depict together. Avoid repetition and avoid listing each image separately.
```

### VHaLM Dataset Stage 2

- **Input:** COCO image triplets (3 images × 5 captions each)
- **Output:** unified caption (≈ 50 tokens)
- **Fields:** group_id, image_ids, captions, merged_caption
- **Usage:** input for Model Stage 1 (Description → VHaLM Training)
