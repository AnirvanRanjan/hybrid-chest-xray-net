# Hybrid Chest X-Ray Net

Multi-label classification of **14 chest pathologies** from NIH Chest X-ray images using a hybrid **DenseNet121 + Swin Transformer** architecture with Feature Pyramid Networks (FPN).

## Architecture

```
Input Image (224x224)
       │
  DenseNet121 (Backbone)
       │
  Multi-scale features (p3, p4, p5)
       │
  Feature Pyramid Network (FPN)
       │
  Swin Transformer (Global Context)
       │
  Classifier Head → 14 disease logits
```

## 14 Disease Classes

| Class | Class | Class | Class |
|-------|-------|-------|-------|
| Atelectasis | Cardiomegaly | Effusion | Infiltration |
| Mass | Nodule | Pneumonia | Pneumothorax |
| Consolidation | Edema | Emphysema | Fibrosis |
| Pleural Thickening | Hernia | | |

## Key Features

- **Hybrid Architecture**: DenseNet121 CNN backbone fused with Swin Transformer via FPN for both local and global feature extraction
- **Asymmetric Loss (ASL)**: Aggressively down-weights easy negatives to handle class imbalance
- **CLAHE Preprocessing**: Contrast-limited adaptive histogram equalization for enhanced image quality
- **Progressive Training**: Frozen backbone → unfrozen transformer → full fine-tuning
- **Multi-scale TTA**: Test-time augmentation with horizontal flip and multi-resolution (320px, 352px) inputs
- **Per-class Threshold Optimization**: Maximizes F1-score for each disease independently
- **GradCAM Visualization**: Interpretability through heatmaps showing regions driving predictions

## Results

| Disease | AUROC | F1 |
|---------|-------|-----|
| Hernia | 0.9319 | 0.4390 |
| Emphysema | 0.9013 | 0.4694 |
| Pneumothorax | 0.8492 | 0.4662 |
| Cardiomegaly | 0.8476 | 0.3080 |
| Edema | 0.8293 | 0.2217 |
| Effusion | 0.8172 | 0.5171 |
| Fibrosis | 0.7953 | 0.1416 |
| Mass | 0.7734 | 0.3129 |
| Pleural Thickening | 0.7532 | 0.2048 |
| Atelectasis | 0.7387 | 0.3606 |
| Consolidation | 0.7211 | 0.2197 |
| Nodule | 0.7087 | 0.2469 |
| Pneumonia | 0.6703 | 0.0778 |
| Infiltration | 0.6587 | 0.4349 |
| **Mean** | **0.7854** | **0.3158** |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Uses the [NIH Chest X-ray](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) dataset (224x224 resized).

## License

MIT
