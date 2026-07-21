# 🦺 Safety Helmet Detection

A computer-vision web app that detects whether construction workers are wearing safety helmets. Upload a site photo and the app locates every worker, marks helmets and bare heads with bounding boxes, and produces a safety compliance report.

Built on a **custom-trained YOLO object-detection model** — not a pretrained one — fine-tuned on ~7,000 annotated construction images.

## 🚀 Live Demo

👉 [Try the app here](https://kask-tespiti-uyari.streamlit.app)

## 📸 Screenshot

![App screenshot](screenshot.png)

## 🛠️ Tech Stack

- **Python** – core language
- **YOLO (Ultralytics)** – real-time object detection, fine-tuned on custom data
- **PyTorch** – deep-learning backend (training on GPU)
- **Roboflow** – dataset sourcing and YOLO-format export
- **OpenCV / Pillow / NumPy** – image processing
- **Streamlit** – web interface and deployment

## ✨ Features

- **Object detection, not just classification:** locates each worker and draws bounding boxes — telling you *what*, *where* and *how many*.
- **Safety report:** counts helmeted vs bare-headed workers, computes a compliance rate, and raises a clear warning when someone is not wearing a helmet.
- **Adjustable confidence threshold:** users can tune detection sensitivity from the interface.
- **Honest scope:** the result is presented as an AI pre-assessment, explicitly not a substitute for an official safety inspection.

## 📊 Model & Training

The model is a YOLO11n network fine-tuned via transfer learning on the **Hard Hat Workers** dataset (~7,000 images, 3 classes: `helmet`, `head`, `person`), trained for 25 epochs on a Tesla T4 GPU.

Evaluation on the held-out validation set:

| Class | mAP@50 |
|-------|--------|
| **helmet** | **0.98** |
| **head (no helmet)** | **0.96** |
| person | 0.03 |

The core task — distinguishing helmeted from bare heads — reaches **96–98% mAP**. The `person` class scores poorly, which class-level analysis traced to severe data scarcity (only ~100 labeled instances vs thousands for the other classes): a clear case where low sample count, not the model, limits performance. It was left as-is since it is not the app's objective.

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Data Source

Hard Hat Workers dataset (Roboflow Universe) – annotated construction-site images.

## ⚠️ Note

This tool provides an AI-based visual pre-assessment and does not replace an official occupational safety inspection.

## 👤 Author

Yiğit Efe USTA – Computer Engineering Student
