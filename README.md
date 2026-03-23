# Vehicle Damage Detection

A deep learning project that classifies vehicle damage from images into 6 categories — built by progressively training and comparing 4 model architectures, with hyperparameter tuning via **Optuna**, and deployed as a live **Streamlit** web app.

---

## Live Demo

Upload a car image and get the damage classification instantly.

![App Screenshot](app_screenshot.png)

---

## Problem Statement

Given an image of a vehicle, classify the damage into one of 6 categories:

| Class | Description |
|---|---|
| `F_Breakage` | Front breakage |
| `F_Crushed` | Front crushed |
| `F_Normal` | Front — no damage |
| `R_Breakage` | Rear breakage |
| `R_Crushed` | Rear crushed |
| `R_Normal` | Rear — no damage |

---

## Model Progression

This project takes a structured approach — training 4 models in increasing order of sophistication to understand the impact of each design decision:

### Model 1 — Custom CNN
A baseline CNN built from scratch with Conv2D, ReLU, and MaxPool layers. Establishes the performance floor.

### Model 2 — CNN with Regularization
Same architecture with **BatchNorm** and **weight decay** (L2 regularization) added to reduce overfitting.

### Model 3 — Transfer Learning with EfficientNet-B0
Pretrained EfficientNet-B0 with all layers frozen except the custom classification head. Significant accuracy jump from leveraging ImageNet features.

### Model 4 — Transfer Learning with ResNet50 + Optuna HPO ✅ (Final Model)
Pretrained ResNet50 with:
- All layers frozen except **layer4** and the FC layer (partial fine-tuning)
- Custom classification head with tunable dropout
- **Optuna** hyperparameter optimisation over 20 trials to find optimal `learning_rate` and `dropout_rate`
- Best params: `dropout_rate = 0.407`, `lr` tuned via log-scale search between 1e-5 and 1e-2

---

## Tech Stack

| Component | Tool |
|---|---|
| Deep Learning | PyTorch |
| Pretrained Models | torchvision (ResNet50, EfficientNet-B0) |
| Hyperparameter Tuning | Optuna |
| Data Augmentation | torchvision transforms |
| Frontend | Streamlit |
| Device Support | MPS / CUDA / CPU |

---

## Data Augmentation

To improve generalisation, training images are augmented with:
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness + contrast ±0.2)
- Resize to 224×224
- ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

## Key Design Decisions

**Why freeze all layers except layer4 in ResNet50?**
Early layers of ResNet learn generic features (edges, textures) that transfer well across domains. Layer4 learns high-level semantic features more specific to the task — unfreezing it allows the model to adapt to vehicle damage patterns while retaining low-level knowledge.

**Why Optuna for hyperparameter tuning?**
Manual grid search is expensive. Optuna uses Tree-structured Parzen Estimator (TPE) to intelligently sample the search space, finding better hyperparameters in fewer trials compared to random or grid search.

**Why CrossEntropyLoss over BCELoss?**
This is a 6-class classification problem. CrossEntropyLoss handles multi-class problems natively, combining log-softmax and negative log-likelihood in one step.

---

## Project Structure

```
vehicle-damage-detection/
│
├── damage_prediction.ipynb     # Model 1–4 training and comparison
├── hyperparam_tuning.ipynb     # Optuna HPO for ResNet50
├── app.py                      # Streamlit web app
├── model_helper.py             # Model loading and inference pipeline
├── model/
│   └── saved_model.pth         # Trained ResNet50 weights
├── dataset/                    # Training images (ImageFolder structure)
├── app_screenshot.png
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/MainHuAj/vehicle-damage-detection.git
cd vehicle-damage-detection

pip install torch torchvision streamlit pillow optuna

streamlit run app.py
```

---

## Author

**Abhinav Bhatera**
[LinkedIn](https://www.linkedin.com/in/abhinav-bhatera) · [GitHub](https://github.com/MainHuAj)
