# YOLOv8 Vehicle Detection
Fine-tuned YOLOv8m on a 7-class vehicle detection dataset using Google Colab (T4 GPU).
Detects vehicles in real-world images with bounding boxes, confidence scores, and per-class metrics.

---

## 📌 Project Overview

| | |
|---|---|
| **Model** | YOLOv8m (pretrained on COCO, fine-tuned) |
| **Dataset** | [Vehicle Detection Dataset](https://www.kaggle.com/datasets/daudshah/vehicle-detection-dataset) — Kaggle |
| **Classes** | `bicycle` · `bus` · `car` · `motorbike` · `rickshaw` · `truck` · `van` |
| **Training** | 27 epochs · image size 640 · batch size 8 · T4 GPU |
| **Framework** | Ultralytics · PyTorch · Supervision |

---

## 🏗️ Pipeline

```
Dataset (Kaggle)
      ↓
  EDA & Class Distribution
      ↓
  YAML Config Setup
      ↓
  YOLOv8m Fine-tuning (27 epochs)
      ↓
  Validation & Test Evaluation (mAP@50, mAP@50-95, Precision, Recall)
      ↓
  Per-class AP breakdown + Visual Inference on test images
```

---

## 📊 Results

| Metric | Validation | Test |
|---|---|---|
| mAP@50 | — | — |
| mAP@50-95 | — | — |
| Precision | — | — |
| Recall | — | — |

> Fill in your numbers from the notebook output after running.

**Per-class AP@50 (Validation):**

| Class | AP@50 |
|---|---|
| bicycle | — |
| bus | — |
| car | — |
| motorbike | — |
| rickshaw | — |
| truck | — |
| van | — |

---

## 🗂️ Repository Structure

```
yolov8-vehicle-detection/
├── yolov8_vehicle_detection.ipynb   # Main Colab notebook (full pipeline)
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 📦 Dependencies

```
ultralytics
supervision
torch
torchvision
Pillow
matplotlib
PyYAML
kaggle
```

---

## 📁 Dataset

Downloaded automatically via the Kaggle API in the notebook.  
Manual download: [Vehicle Detection Dataset on Kaggle](https://www.kaggle.com/datasets/daudshah/vehicle-detection-dataset)

Dataset split:
- `train/` — training images + YOLO-format labels
- `valid/` — validation set
- `test/`  — held-out test set

---

## 🧠 Key Techniques

- **Transfer Learning** — YOLOv8m pretrained on COCO (80 classes), fine-tuned on 7 vehicle classes
- **Early Stopping** — patience=10 to prevent overfitting
- **EDA** — class distribution plots across train/val splits before training
- **Evaluation** — mAP@50, mAP@50-95, precision, recall on both val and test sets
- **Visual Inference** — bounding box overlays on random test images

---

## 📈 Training Curves

Training curves (loss, mAP per epoch) are auto-saved by Ultralytics to `runs/detect/vehicle_detector/`.  
They're displayed inline in the notebook after training completes.

---

## 🔧 Customisation

To train with a different YOLOv8 variant, change Cell 9:

```python
model = YOLO('yolov8n.pt')   # nano  — fastest
model = YOLO('yolov8m.pt')   # medium — recommended ✓
model = YOLO('yolov8l.pt')   # large  — most accurate
```

To adjust training:

```python
results = model.train(
    epochs=50,      # more epochs
    batch=16,       # larger batch (needs more VRAM)
    imgsz=1280,     # higher resolution
)
```

---

## 🙋 Author

**B Sai Preeti**  
