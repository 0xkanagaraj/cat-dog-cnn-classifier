# Dog–Cat Image Classification with a 3-Layer CNN

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images as **cat** or **dog**. The model uses three convolutional layers, data augmentation and dropout, and is trained end to end in a single Google Colab notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0xkanagaraj/cat-dog-cnn-classifier/blob/main/cat_dog_cnn.ipynb)

## Overview

Image classification is a core computer vision task, and distinguishing cats from dogs is a classic benchmark. This project builds a compact CNN from scratch (no pretrained weights) and covers the full workflow: loading data, augmenting images, training, evaluating and predicting on new photos.

## Dataset

The notebook loads the **cats_vs_dogs** dataset from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs), which contains about 23,000 labeled cat and dog images. It is downloaded automatically on the first run (a few minutes), so no manual download or Kaggle login is needed.

| Split | Share of data |
|-------|---------------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Images are resized to 150 × 150 pixels.

## Repository Structure

```
cat-dog-cnn-classifier/
├── cat_dog_cnn.ipynb   # Colab notebook with the full code
└── README.md
```

## Getting Started

### Run in Google Colab

1. Click the **Open In Colab** badge above.
2. Select **Runtime → Change runtime type → GPU** (T4) for faster training.
3. Choose **Runtime → Run all**.

### Run locally

```bash
git clone https://github.com/0xkanagaraj/cat-dog-cnn-classifier.git
cd cat-dog-cnn-classifier
pip install numpy matplotlib scikit-learn tensorflow tensorflow-datasets jupyter
jupyter notebook cat_dog_cnn.ipynb
```

The last notebook cell (upload your own image) uses `google.colab` and only works in Colab.

## Model Architecture

| Stage | Layer | Details |
|-------|-------|---------|
| Input | Image | 150 × 150 × 3 |
| Preprocessing | Data augmentation | Random flip, rotation, zoom (training only) |
| Preprocessing | Rescaling | Pixels scaled to [0, 1] |
| Conv block 1 | Conv2D + MaxPooling | 32 filters, 3×3, ReLU |
| Conv block 2 | Conv2D + MaxPooling | 64 filters, 3×3, ReLU |
| Conv block 3 | Conv2D + MaxPooling | 128 filters, 3×3, ReLU |
| Classifier | Flatten → Dropout (0.5) → Dense | 128 units, ReLU |
| Output | Dense | 1 unit, sigmoid |

| Setting | Value |
|---------|-------|
| Optimizer | Adam (learning rate 0.001) |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Max epochs | 40 |
| Callbacks | Early stopping (patience 6), ReduceLROnPlateau |

## Workflow

1. Download the dataset with TensorFlow Datasets and split it 70/15/15
2. Resize images and build cached, prefetched input pipelines
3. Visualize sample images
4. Build and compile the CNN
5. Train with early stopping and learning-rate reduction
6. Plot accuracy and loss curves
7. Evaluate on the test set (accuracy, classification report, confusion matrix)
8. View sample predictions, save the model and test your own images

## Requirements

- Python 3.8+
- tensorflow
- tensorflow-datasets
- numpy
- matplotlib
- scikit-learn

All of these are preinstalled in Google Colab.

## Results

Add your final test accuracy and plots here after running the notebook, for example:

| Metric | Value |
|--------|-------|
| Test accuracy | _your value_ |
| Test loss | _your value_ |

## Possible Extensions

- Add batch normalization and more filters
- Use transfer learning (MobileNetV2, ResNet50, EfficientNet) and compare
- Visualize feature maps or Grad-CAM heatmaps
- Deploy as a small web app with Gradio or Streamlit

## Author

[0xkanagaraj](https://github.com/0xkanagaraj)
