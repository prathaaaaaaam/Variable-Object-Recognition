# 🎯 Variable Object Recognition

> Real-time object detection using YOLOv5 & YOLOv8 with live webcam feed, bounding box visualization, and confidence scoring — powered by PyTorch and OpenCV.

**Made by [Pratham Sorte](https://github.com/prathaaaaaaam)**

---

## 📌 Table of Contents

- [About](#-about)
- [Demo](#-demo)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Training](#-training)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 About

**Variable Object Recognition** is a real-time computer vision project that leverages the power of YOLO (You Only Look Once) deep learning models — both YOLOv5 and YOLOv8 — to detect and classify multiple objects from a live webcam stream.

The system captures video frames, runs inference through a pre-trained or custom-trained YOLO model, and overlays bounding boxes along with class labels and confidence scores directly on the video feed. It supports training custom models and experimenting across multiple training runs.

This project was built to explore object detection pipelines from data preparation to real-time deployment.

---

## ✨ Features

- 🎥 **Live webcam object detection** with real-time bounding boxes
- 🏷️ **Class label + confidence score** overlay on each detected object
- ⚡ **YOLOv5 & YOLOv8** model support (switchable)
- 🗂️ **Multiple training runs** stored for comparison (`train`, `train2`, `train3`, `train4`)
- 📦 Pre-trained `yolov8n.pt` model included for instant inference
- 🧪 COCO8 dataset integration for training and validation
- 🖥️ Simple single-file Python scripts for quick experimentation

---

## 📁 Project Structure

```
Variable-Object-Recognition/
│
├── YOLOV5/                   # YOLOv5 model files and configs
├── yolov8-silva-main/        # YOLOv8 implementation/reference
│
├── datasets/
│   └── coco8/                # COCO8 dataset (8-class subset for training)
│
├── train/                    # Training run 1 — weights, logs, metrics
├── train2/                   # Training run 2
├── train3/                   # Training run 3
├── train4/                   # Training run 4
│
├── utils/                    # Utility scripts and helpers
├── __pycache__/              # Python bytecode cache
│
├── ex.py                     # Main real-time detection script (YOLOv8)
├── exp.py                    # Experimental/alternate detection script
├── yolov8n.pt                # Pre-trained YOLOv8 Nano model weights
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core language |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection model |
| [YOLOv5](https://github.com/ultralytics/yolov5) | Alternative detection model |
| OpenCV (`cv2`) | Video capture & frame rendering |
| PyTorch | Deep learning backend |
| COCO8 Dataset | Training & validation data |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/prathaaaaaaam/Variable-Object-Recognition.git
cd Variable-Object-Recognition
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install ultralytics opencv-python torch torchvision
```

> **Note:** If you have a CUDA-capable GPU, install the GPU version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for faster inference.

---

## 🚀 Usage

### Run real-time object detection (webcam)

```bash
python ex.py
```

- The webcam will open and start detecting objects in real time.
- Detected objects are highlighted with **blue bounding boxes**.
- **Class name** and **confidence score** appear above each box.
- Press **`Q`** to quit the detection window.

### Switch models

In `ex.py`, change the model path on line 7:

```python
# Use the default pre-trained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Or use your own custom-trained model from a training run
model = YOLO("train/weights/best.pt")
```

---

## 📊 Dataset

This project uses the **COCO8** dataset — a compact 8-image subset of the COCO dataset ideal for quick training experiments and pipeline testing.

- Located at: `datasets/coco8/`
- Contains images with standard COCO class annotations
- Used for training, validation, and benchmarking the custom model

---

## 🏋️ Training

To train your own model on the COCO8 dataset (or any custom dataset):

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load base model

model.train(
    data="datasets/coco8/coco8.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="train5"
)
```

Training results (weights, confusion matrix, loss curves) are saved automatically to the specified `name` folder.

---

## 📈 Results

Multiple training runs are stored in this repository:

| Run | Folder | Notes |
|---|---|---|
| Run 1 | `train/` | Initial baseline training |
| Run 2 | `train2/` | Hyperparameter tuning |
| Run 3 | `train3/` | Adjusted batch size / epochs |
| Run 4 | `train4/` | Latest/best run |

Each folder contains:
- `weights/best.pt` — best model checkpoint
- `weights/last.pt` — last epoch checkpoint
- Training metrics (loss curves, mAP, confusion matrix)

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute it.

---

## 👨‍💻 Author

**Pratham Sorte**
- GitHub: [@prathaaaaaaam](https://github.com/prathaaaaaaam)

---

## 🙌 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 framework
- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [COCO Dataset](https://cocodataset.org/) for benchmark data
- OpenCV community for the video processing tools
