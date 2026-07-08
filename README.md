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

```
                         ┌─────────────────────┐
                         │   Input Image 224×224 │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      DenseNet121 Backbone      │
                    │   (Pre-trained on ImageNet)    │
                    └───────┬───────┬───────┬───────┘
                            │       │       │
                           p3      p4      p5
                            │       │       │
                    ┌───────▼───────▼───────▼───────┐
                    │   Feature Pyramid Network     │
                    │         (FPN Fusion)          │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │    Swin Transformer (Swin-B)  │
                    │     (Global Context)          │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   Classifier Head (14 Class)  │
                    │  LayerNorm → Linear → GELU    │
                    │  → Dropout → Linear           │
                    └───────────────────────────────┘
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

| | | | |
|:---:|:---:|:---:|:---:|
| Atelectasis | Cardiomegaly | Effusion | Infiltration |
| Mass | Nodule | Pneumonia | Pneumothorax |
| Consolidation | Edema | Emphysema | Fibrosis |
| Pleural Thickening | Hernia | | |

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
