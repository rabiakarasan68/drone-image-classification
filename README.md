# Drone Image Classification

An image classification project developed using **Python, TensorFlow/Keras, and MobileNetV2**. The model uses Transfer Learning to classify images into four categories: **Drone, Bird, Airplane, and Sky**.

## Project Overview

The main objective of this project is to explore deep learning and image classification techniques using a pre-trained MobileNetV2 model.

The project includes:
- Image preprocessing
- Data augmentation
- Transfer Learning with MobileNetV2
- Fine-tuning the last 20 layers
- Model training and evaluation
- Image prediction using OpenCV

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure

```text
drone-image-classification/
│
├── dataset/
│   ├── train/
│   │   ├── airplane/
│   │   ├── bird/
│   │   ├── drone/
│   │   └── sky/
│   │
│   └── test/
│       ├── airplane/
│       ├── bird/
│       ├── drone/
│       └── sky/
│
├── download_dataset.py
├── rename_all.py
├── split_dataset.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

## Dataset

The dataset is organized into four classes:

| Class | Description |
|---|---|
| Drone | Images of drones |
| Bird | Images of birds |
| Airplane | Images of airplanes |
| Sky | Images of the sky |

**Dataset note:** The current `download_dataset.py` script uses one image URL for each of the Bird, Airplane, and Sky classes and repeats that same URL 50 times. Therefore, it does not download 50 unique images per class. The script also does not include the Drone class. It is a basic download script and should not be considered a complete dataset collection solution.

The Drone class and a suitable, varied dataset must be provided separately for the complete four-class training process.

The dataset is not included in this repository.

## Model Architecture

The project uses **MobileNetV2**, a convolutional neural network pre-trained on ImageNet.

The original classification head is removed and replaced with custom layers:

1. MobileNetV2 feature extraction
2. Global Average Pooling
3. Dense layer with 128 neurons and ReLU activation
4. Softmax output layer for four-class classification

### Training Configuration

| Parameter | Value |
|---|---|
| Input Image Size | 224 × 224 |
| Batch Size | 16 |
| Epochs | 30 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Fine-tuning | Last 20 layers |
| Validation Split | 20% |

Data augmentation techniques include rotation, shifting, zooming, horizontal flipping, and brightness adjustment.

## Installation

Clone the repository:

```bash
git clone https://github.com/rabiakarasan68/drone-image-classification.git
```

Navigate to the project directory:

```bash
cd drone-image-classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Place the dataset in the appropriate folders before running the scripts.

## Usage

### 1. Prepare the Dataset

Organize the images into class-specific folders under `dataset/train`.

### 2. Rename Images

```bash
python rename_all.py
```

### 3. Split the Dataset

```bash
python split_dataset.py
```

This script moves approximately 20% of the images from each training class into the corresponding test folder.

**Note:** The script moves files rather than copying them. Run it only once on the original dataset.

### 4. Train the Model

```bash
python train.py
```

The trained model is saved as:

```text
drone_model.h5
```

### 5. Evaluate the Model

```bash
python evaluate.py
```

The evaluation script calculates:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 6. Make Predictions

```bash
python predict.py
```

The prediction script classifies test images and displays the prediction results using OpenCV.

## Important Note

This project performs **image classification**, not object detection. It predicts the main class of an input image and does not locate objects or generate bounding boxes.

Model performance should be evaluated using the test dataset. No accuracy value is reported here because a verified evaluation result has not yet been added.

GitHub: [@rabiakarasan68](https://github.com/rabiakarasan68)
