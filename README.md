# 🍅 Tomato Disease Classification using EfficientNetB0

A deep learning model for automated detection and classification of tomato leaf diseases using transfer learning with EfficientNetB0.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Per-Class Performance](#per-class-performance)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Training Details](#training-details)

---

## Overview

This project classifies tomato leaf images into **11 disease categories** (including healthy) using a fine-tuned EfficientNetB0 model. It achieves a **92.1% test accuracy** across 6,683 test images.

The model can assist farmers and agricultural professionals in early identification of tomato diseases, enabling timely treatment and reducing crop loss.

---

## Dataset

The dataset contains labelled images organized into `train/` and `valid/` directories, covering **11 classes**:

| Class | Description |
|---|---|
| Bacterial_spot | Bacterial infection causing dark leaf spots |
| Early_blight | Fungal disease with concentric ring lesions |
| Late_blight | Water mold causing irregular necrotic lesions |
| Leaf_Mold | Fungal mold on leaf surface |
| Septoria_leaf_spot | Small circular spots with dark borders |
| Spider_mites Two-spotted_spider_mite | Pest infestation causing stippled leaves |
| Target_Spot | Circular lesions with target-like rings |
| Tomato_Yellow_Leaf_Curl_Virus | Viral disease causing leaf curl and yellowing |
| Tomato_mosaic_virus | Viral mosaic pattern on leaves |
| healthy | Healthy tomato leaf |
| powdery_mildew | White powdery fungal coating on leaves |

**Dataset split:**
- Training: 80% of training folder (stratified)
- Validation: 20% of training folder (stratified)
- Test: Full `valid/` folder — **6,683 images**

---

## Model Architecture

Built on **EfficientNetB0** pre-trained on ImageNet with a custom classification head:

```
EfficientNetB0 (frozen, ImageNet weights)
    └── GlobalAveragePooling2D
    └── BatchNormalization
    └── Dense(512, relu)
    └── Dropout(0.4)
    └── Dense(256, relu)
    └── Dropout(0.2)
    └── Dense(11, softmax)
```

**Training configuration:**
- Optimizer: Adam (`lr=0.0005`)
- Loss: Categorical Crossentropy
- Epochs: 15
- Batch size: 64
- Input size: 224×224×3
- Callbacks: `ReduceLROnPlateau`, `ModelCheckpoint`

---

## Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **92.14%** |
| Macro Avg Precision | 92.31% |
| Macro Avg Recall | 92.25% |
| Macro Avg F1-Score | 92.14% |

### Training Curves

The model converges smoothly with training and validation accuracy both reaching ~92% by epoch 14, with no significant overfitting observed.

---

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Bacterial_spot | 97.1% | 91.7% | 94.3% | 732 |
| Early_blight | 89.6% | 85.4% | 87.4% | 643 |
| Late_blight | 92.0% | 93.3% | 92.7% | 792 |
| Leaf_Mold | 94.8% | 93.2% | 94.0% | 739 |
| Septoria_leaf_spot | 87.8% | 87.7% | 87.7% | 746 |
| Spider_mites Two-spotted | 91.9% | 91.9% | 91.9% | 435 |
| Target_Spot | 74.6% | 95.2% | 83.7% | 457 |
| Tomato_Yellow_Leaf_Curl_Virus | 99.6% | 95.8% | 97.6% | 498 |
| Tomato_mosaic_virus | 96.6% | 92.8% | 94.7% | 584 |
| healthy | 95.9% | 95.7% | 95.8% | 805 |
| powdery_mildew | 95.5% | 92.1% | 93.7% | 252 |

> **Note:** `Target_Spot` has the lowest precision (74.6%), showing some confusion with visually similar classes like Spider mites and healthy leaves. `Tomato_Yellow_Leaf_Curl_Virus` achieves the highest precision at 99.6%.

---

## Project Structure

```
tomato-disease-classification/
│
├── mptomatodisease.ipynb          # Main Jupyter notebook
├── best_tomato_model.h5           # Saved best model weights
│
├── outputs/
│   ├── training_history.png       # Accuracy & loss curves
│   ├── confusion_matrix.png       # Confusion matrix on test set
│   ├── sample_predictions.png     # Sample prediction visualizations
│   ├── classification_report.csv  # Per-class metrics
│   └── test_predictions.csv       # Full test set predictions
```

---

## Installation & Requirements

```bash
pip install tensorflow keras scikit-learn pandas matplotlib seaborn
```

**Key dependencies:**
- Python 3.8+
- TensorFlow / Keras 2.x
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

---

## Usage

1. **Clone / open the notebook** in Kaggle or a local Jupyter environment.
2. **Attach the tomato disease dataset** (ensure it has `train/` and `valid/` subdirectories).
3. **Run all cells** — the notebook will:
   - Audit and visualize the dataset
   - Build and train the EfficientNetB0 model
   - Evaluate on the test set
   - Generate confusion matrix, classification report, and sample predictions
4. The best model is saved automatically as `best_tomato_model.h5`.

**For inference on new images:**
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image

model = load_model('best_tomato_model.h5')

img = Image.open('your_leaf_image.jpg').resize((224, 224))
x = preprocess_input(np.expand_dims(np.array(img), axis=0))
pred = model.predict(x)
print(class_labels[np.argmax(pred)])
```

---

## Training Details

- **Data augmentation** (training only): rotation (±30°), width/height shift (±20%), shear, zoom, horizontal flip
- **Preprocessing**: EfficientNet-specific `preprocess_input` (scales pixels to the expected range)
- **Learning rate scheduler**: `ReduceLROnPlateau` reduces LR by 10× if validation loss plateaus for 2 epochs (min LR: 1e-7)
- **Best model checkpoint**: saved based on best `val_accuracy`

---

## License

This project is for educational and research purposes. The dataset used is sourced from https://www.kaggle.com/datasets/ashishmotwani/tomato/data
