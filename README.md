# **Penguin Classifier (CS3120 Final Project)**

This repository contains a Python-based image classification model that uses a convolutional neural network (CNN) to determine whether an input image contains a penguin. The project was developed as part of the CS3120 course at MSU Denver in Spring 2025.

## **Features**

**Overview**
  - Binary image classification using a custom-trained CNN
  - Preprocessing pipeline for image resizing and normalization
  - Interactive demo folder for classifying new user-supplied images
  - Automatically saved models, training plots, and evaluation metrics

**Model Design**
  - Convolutional layers with ReLU activations
  - MaxPooling layers to reduce spatial dimensions
  - Dropout and BatchNormalization for generalization and stability
  - Fully connected (dense) classification output with Softmax

**Visualization & Evaluation**
  - Training accuracy and loss plots (auto-generated)
  - Confusion matrix for performance analysis
  - Classification report with precision, recall, and F1-score

**Demo Tools**
  - `demo_input/`: drop in any `.jpg`, `.jpeg`, or `.png` file to test the model
  - Outputs include predicted class and confidence score

## **Tech Stack**

  - Python 3.x
  - TensorFlow / Keras – CNN model training and inference
  - NumPy – numerical computation
  - OpenCV – image classification in demo mode
  - Matplotlib – training visualizations
  - Scikit-learn – evaluation metrics and data splitting
  - Jupyter Notebook – development environment

## **Usage**

To run the project locally:

1. Clone the repository:  
   `git clone https://github.com/Jerem-Dough/CNN-Penguin-Classifier.git`
2. Create and activate Python environment `python -m venv`, then `pip install -r requirements.txt`
3. Open `penguin_classifier.ipynb` in Jupyter Notebook
4. Ensure your dataset is placed in `../data/penguin/` and `../data/not_penguin/`
5. Run all cells to train and evaluate the model
6. Add custom images to `../demo_input/` and run the final cell block to classify them

