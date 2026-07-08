<div align="center">

# Hybrid Chest X-Ray Net

**Multi-label classification of 14 chest pathologies from X-ray images**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-T4%20GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

A hybrid **DenseNet121 + Swin Transformer** model with Feature Pyramid Networks for detecting multiple diseases in chest X-rays.

```
Mean AUROC: 0.7854 │ 14 Diseases │ Multi-Label Classification
```

---

</div>

## Architecture

<div align="center">

```mermaid
flowchart TD
    A["<b>Input Image</b><br/>224 × 224 × 3"] --> B

    B["<b>DenseNet121 Backbone</b><br/>Pre-trained on ImageNet<br/>─────────────────<br/>conv0 → norm0 → relu0 → pool0<br/>denseblock1 → transition1<br/>denseblock2 → transition2<br/>denseblock3 → transition3<br/>denseblock4 → norm5"]
    B -->|"p3 (256 ch)"| C
    B -->|"p4 (512 ch)"| C
    B -->|"p5 (1024 ch)"| C

    C["<b>Feature Pyramid Network</b><br/>─────────────────<br/>1×1 Conv projection<br/>Top-down pathway<br/>Lateral connections<br/>→ Fused 256-ch feature map"]
    C --> D

    D["<b>Swin Transformer (Swin-B)</b><br/>─────────────────<br/>Window-based self-attention<br/>Shifted window layers<br/>Global context modeling"]
    D --> E

    E["<b>Classification Head</b><br/>─────────────────<br/>LayerNorm<br/>Linear (1024 → 512)<br/>GELU Activation<br/>Dropout (0.5)<br/>Linear (512 → 14)"]
    E --> F["<b>14 Disease Probabilities</b><br/>Sigmoid → Multi-label Prediction"]

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    style B fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#bf360c
    style C fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style D fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#b71c1c
    style E fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c
    style F fill:#e0f7fa,stroke:#00695c,stroke-width:2px,color:#004d40
```

</div>

---

## Sample Predictions

> Add your GradCAM and prediction images here after running the notebook.
>
> ```bash
> # Save outputs from notebook:
> # - learning_history.png
> # - pr_curves.png  
> # - confusion_matrices.png
> # Then place them in an assets/ folder and reference below:
> ```

<div align="center">

<!-- Add your images like this: -->
<!-- ![GradCAM Results](assets/gradcam_results.png) -->
<!-- ![Learning History](assets/learning_history.png) -->

*Run the notebook to generate visualization outputs*

</div>

---

## 14 Disease Classes

<div align="center">

| # | Disease | | # | Disease |
|:-:|:--------|:-:|:-:|:--------|
| 1 | Atelectasis | | 8 | Pneumothorax |
| 2 | Cardiomegaly | | 9 | Consolidation |
| 3 | Effusion | | 10 | Edema |
| 4 | Infiltration | | 11 | Emphysema |
| 5 | Mass | | 12 | Fibrosis |
| 6 | Nodule | | 13 | Pleural Thickening |
| 7 | Pneumonia | | 14 | Hernia |

</div>

---

## Results

<div align="center">

| Disease | AUROC | F1 Score | |
|:--------|------:|---------:|:-:|
| **Hernia** | **0.9319** | 0.4390 | Best |
| **Emphysema** | **0.9013** | **0.4694** | |
| **Pneumothorax** | **0.8492** | 0.4662 | |
| **Cardiomegaly** | **0.8476** | 0.3080 | |
| **Edema** | 0.8293 | 0.2217 | |
| **Effusion** | 0.8172 | **0.5171** | Best F1 |
| **Fibrosis** | 0.7953 | 0.1416 | |
| **Mass** | 0.7734 | 0.3129 | |
| **Pleural Thickening** | 0.7532 | 0.2048 | |
| **Atelectasis** | 0.7387 | 0.3606 | |
| **Consolidation** | 0.7211 | 0.2197 | |
| **Nodule** | 0.7087 | 0.2469 | |
| **Pneumonia** | 0.6703 | 0.0778 | Hardest |
| **Infiltration** | 0.6587 | 0.4349 | |
| **Mean** | **0.7854** | **0.3158** | |

</div>

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Architecture** | DenseNet121 CNN + Swin Transformer fused via FPN |
| **Asymmetric Loss** | Down-weights easy negatives for imbalanced data |
| **CLAHE Preprocessing** | Adaptive histogram equalization for better contrast |
| **Progressive Training** | Freeze → Unfreeze → Full fine-tuning |
| **Multi-scale TTA** | Horizontal flip + multi-resolution inference |
| **Threshold Optimization** | Per-class F1-maximized decision boundaries |
| **GradCAM** | Visual explanations for model predictions |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Kartiken07/hybrid-chest-xray-net.git
cd hybrid-chest-xray-net

# Install dependencies
pip install -r requirements.txt
```

### Run on Kaggle

1. Upload `final - Copy.ipynb` to Kaggle
2. Enable **GPU** (T4) in notebook settings
3. Add the [NIH Chest X-ray dataset](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) as data source
4. Run all cells

---

## Dataset

<div align="center">

[![Dataset](https://img.shields.io/badge/NIH_Chest_X--ray-14_Classes-FF6F00?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)

</div>

---

<div align="center">

**Built with PyTorch**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

</div>
