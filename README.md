HEAD
### \# Face Recognition Project

### 

### This project implements a face recognition tool using PyTorch.

### It uses a dataset of South Indian celebrity photos for training and testing.

### 

### \## How to run

### 1\. Install required libraries (Torch, Torchvision, matplotlib, etc.)

### 2\. Place your images in the dataset folder ("south-indian-celebrity-dataset/")

### 3\. Run python-code.py

### 

### \## Dataset:

### -Source: kagglehub.dataset\_download("gunarakulangr/south-indian-celebrity-dataset")

### \- Format: Each celebrity is a separate folder containing their images

### 

### \## Results

### \- Shows training accuracy, loss, and visualizes predictions

### 

### \## Author

### Kaviya M

### 

# Face-Recognition-mode
This project implements a face recognition tool using PyTorch.
It uses a dataset of South Indian celebrity photos for training and testing.

## How to run
1. Install required libraries (Torch, Torchvision, matplotlib, etc.)
2. Place your images in the dataset folder ("south-indian-celebrity-dataset/")
3. Run python-code.py

## Dataset:
-Source: kagglehub.dataset_download("gunarakulangr/south-indian-celebrity-dataset")
- Format: Each celebrity is a separate folder containing their images

## Model Architecture Choice:
ResNet-18 as my model architecture.It is known for:
- Good performance on image classification tasks, even with limited data.  
- Its use of residual (skip) connections, which help avoid the vanishing gradient problem and allow deeper networks to train effectively.  
- Being lightweight enough to run on standard hardware while being powerful enough to learn complex face features.  
This architecture is a strong default option for face recognition projects because it strikes a good balance between accuracy and computational efficiency.  

## Preprocessing and Data Augmentation:
- Resizing: All images are resized to a fixed dimension (128x128) to ensure a uniform input shape for the model.  
- Random Horizontal Flip: This randomly flips images during training, exposing the model to mirrored versions of faces to improve generalization.  
- Normalization: Each pixel is adjusted to have values between -1 and 1, which helps the learning algorithm converge more effectively.  

## Reasoning:  
-Resizing and normalization are vital for consistent, stable learning.  
-Flipping increases data variability, which is important when the dataset is small or lacks diversity.  

## Interpretation of Model Performance:
From my results:  
-Loss consistently decreases and training accuracy increases across 5 epochs, as shown in the plots and output. This indicates that the model is successfully learning to fit the training data.  
-Test accuracy is modest(~0.17), which is much slightly lower than training accuracy.  

## Strengths:  
-Training metrics (loss and accuracy) improve, demonstrating that the model can recognize patterns in the training set.  
-Data augmentation (horizontal flip) provides slight benefits in preventing overfitting compared to training without it.  

## Weaknesses:  
-Modest test accuracy suggests possible overfitting. The model performs well on familiar data but struggles to generalize to new, unseen faces.  
-Possible issues include having too few training images per class, highly similar backgrounds, or images lacking diversity in pose or lighting.  

## Results
- Shows training accuracy, loss, and visualizes predictions

## Author
Kaviya M

 0e66919b1bfa00ab2db3ce125c8bdf80d786e41f
 <img width="504" height="230" alt="Screenshot 2025-10-29 203805" src="https://github.com/user-attachments/assets/dd22ce68-2416-4f75-96ef-a072178ba3a8" />
 <img width="1510" height="684" alt="Screenshot 2025-10-29 203702" src="https://github.com/user-attachments/assets/39248e3c-c3ad-406a-952c-3f98fccf2290" />
 <img width="563" height="568" alt="Screenshot 2025-10-29 203926" src="https://github.com/user-attachments/assets/1ebda3b0-ad22-4d6e-b05e-499a0c349246" />



