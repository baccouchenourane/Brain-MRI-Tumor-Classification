#  Brain MRI Tumor Classification

A deep learning project for classifying brain MRI images into four categories using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras.

The project covers the complete machine learning workflow, from image preprocessing and model training to performance evaluation and an interactive web application built with Gradio.

<p align="center">
  <strong>Deep Learning · Computer Vision · Medical Imaging · TensorFlow · Python</strong>
</p>

---

##  Project Overview

The objective of this project is to develop a CNN-based image classification model capable of identifying four categories from brain MRI images:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

The project also includes a Gradio interface that allows users to upload an MRI image and view the model's classification scores.

##  Dataset

The project uses the **Brain Tumor MRI Dataset** by Masoud Nickparvar.

| Dataset split | Number of images |
|---|---:|
| Training | 5,712 |
| Testing | 1,311 |
| **Total** | **7,023** |

### Class distribution

| Class | Training | Testing |
|---|---:|---:|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |

The dataset is organized into separate training and testing directories, each containing four class folders.

##  Data Preprocessing

The following preprocessing steps were applied:

- Resizing images to **224 × 224** pixels.
- Converting images to RGB format.
- Normalizing pixel values to the range [0, 1].
- Creating a validation subset from the training data.
- Using batches of 32 images.
- Applying prefetching to improve data loading efficiency.

##  CNN Architecture

The selected model uses a sequential CNN architecture:

| Layer | Configuration |
|---|---|
| Input | 224 × 224 × 3 |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 128 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | Feature vector |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Output | 4 units, Softmax |

**Training configuration:**
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metric: Accuracy
- Epochs: 10

## 📈 Model Evaluation

The selected CNN achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Test Accuracy | **93.75%** |
| Test Loss | 0.3958 |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Glioma | 0.98 | 0.83 | 0.90 |
| Meningioma | 0.85 | 0.92 | 0.88 |
| No Tumor | 0.95 | 0.99 | 0.97 |
| Pituitary | 0.98 | 1.00 | 0.99 |
| **Macro Average** | **0.94** | **0.93** | **0.93** |
| **Weighted Average** | **0.94** | **0.94** | **0.94** |

These results reflect performance on the selected dataset split. They do not establish clinical reliability or guarantee generalization to images from other sources.

##  Interactive Web Application

The project includes a web interface developed with **Gradio**.

Users can:
1. Upload a brain MRI image.
2. Submit it to the trained CNN model.
3. View the classification scores for all four categories.

The application uses the saved Keras model (`brain_tumor_cnn1.keras`).

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep learning and CNN development |
| NumPy | Numerical operations |
| Pillow | Image processing |
| Gradio | Interactive web interface |
| Google Colab | Model development and training |
| Git & GitHub | Version control and project management |
| Git LFS | Storage of the trained model |

##  Project Structure

```text
Brain-MRI-Tumor-Classification/
│
├── app/
│   └── app.py
│
├── notebooks/
│   └── brain_tumor_classification.ipynb
│
├── brain_tumor_cnn1.keras
├── requirements.txt
├── .gitattributes
└── README.md
```

##  Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/baccouchenourane/Brain-MRI-Tumor-Classification.git

cd Brain-MRI-Tumor-Classification
```

### 2. Download the model using Git LFS

```bash
git lfs install
git lfs pull
```

Make sure the trained model is available in the project root.

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the application

```bash
python app/app.py
```

Open the local Gradio URL displayed in your terminal.

##  Limitations & Responsible Use

This project is developed for **academic and educational purposes only**.

- It is not a certified medical device.
- It must not be used for medical diagnosis or treatment decisions.
- Model scores are not calibrated clinical probabilities.
- Performance may vary on MRI images from different hospitals, scanners, and populations.
- Further validation on independent datasets is necessary before considering any real-world medical application.
## Results

### Training Curves
![Training Curves](screenshots/training_curves.png)

### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

### Gradio Application
![Gradio App](screenshots/gradio_app.png)

### Dataset Examples
![Dataset Examples](screenshots/dataset_examples.png)
##  Author

**Nourane Baccouche**

Software Engineering & Information Systems Student  
Interested in Artificial Intelligence, Data Science and Software Development.

- GitHub: [@baccouchenourane](https://github.com/baccouchenourane)

---

<p align="center">
  <strong>Exploring the intersection of Artificial Intelligence and Medical Imaging.</strong>
</p>
