
# CS3120 Final Project Report - Jeremy Dougherty

## 1. What is the project about?

This project was a hands-on opportunity to design and implement a fully functional convolutional neural network (CNN) for image classification. Rather than solving a real-world problem, the primary goal was to gain experience with core machine learning techniques including data preprocessing, neural network architecture, and performance evaluation. The classifier determines whether an image contains a penguin or not, and serves as a solid introduction to deep learning, image processing, and Python-based model development.

## 2. What is original vs. reused?

### Original Work
- The overall implementation and architecture of the CNN.
- Image preprocessing pipeline including normalization and resizing.
- The interactive demo input folder and automated batch image classification.
- File-saving logic for models, plots, and reports using dynamic timestamps.

### Adapted / Referenced
- Visualization ideas (e.g., confusion matrix, training curves) adapted from online tutorials and ChatGPT suggestions.
- Keras callbacks (`EarlyStopping`, `ReduceLROnPlateau`) and model structure syntax adapted from examples.
- AI was used to debug issues and optimize training performance.

### Cited References
- ChatGPT (https://chatgpt.com/)
- Images.cv (https://images.cv/)
- Kaggle (https://www.kaggle.com/)
- StackOverflow (https://stackoverflow.com/)
- Scikit-learn Documentation (https://scikit-learn.org/)
- TensorFlow/Keras Documentation (https://www.tensorflow.org/)
- Nicholas Renotte (https://www.youtube.com/@NicholasRenotte)

## 3. Libraries and Packages

- `tensorflow`, `keras` – model building, training
- `numpy` – numerical processing
- `PIL` – image loading and resizing
- `cv2` (OpenCV) – demo image input
- `matplotlib.pyplot` – plotting training curves
- `sklearn.model_selection`, `metrics`, `utils` – evaluation and data splitting
- `os`, `glob`, `datetime` – file handling utilities

## 4. Algorithms Used

### Model Type: Convolutional Neural Network (CNN)
CNN was selected because of its effectiveness in analyzing visual patterns within images. My application utilizes:

- `Conv2D` layers to extract local features
- `MaxPooling2D` to reduce dimensionality and emphasize the most relevant pixels
- `BatchNormalization` to improve convergence speed
- `Dropout` to reduce overfitting issues
- `Dense` layers for fully connected classification output
- `Softmax` activation to yield class probabilities for final binary classification

This structure is standard and well-suited for lightweight image classification tasks.

## 5. Hyperparameter Choices

- **Epochs**: After experimenting with values from 10 to 50, I found diminishing returns after ~30 epochs. Validation accuracy plateaued and training time increased with minimal benefit.
- **Batch Size**: A batch size of 32 provided a good balance of training speed and model stability. Lower sizes were unstable, while higher sizes consumed excess memory.
- **Dropout Rates**: Dropout was added at 0.2 and 0.4 respectively to prevent overfitting while keeping the model expressive enough. These values were suggested by AI feedback and confirmed with test runs.
- **Image Size**: One of my penguin datasets used 64x64 images natively, so all images were resized to 64x64 for compatibility. This size also helped reduce memory load and training time without sacrificing accuracy.

## 6. Additional Notes

- A `demo_input` folder allows users to drag and drop new images for quick classification.
- Timestamped model saving ensures results are versioned and organized for re-use.
- Classification results include confidence scores and emojis as label markers.
- Output plots (training curves, confusion matrix) are auto-saved to an `output/` directory for tracking model history.
- Modular classes and a local Jupyter Notebook make the code easily modifiable. 
- **Fun Factor:** 🐧🐧🐧🐧🐧/5
