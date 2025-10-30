# Face Recognition Project

This project implements a face recognition tool using PyTorch. The model is trained to identify South Indian celebrities using a curated dataset of profile photos, showcasing core deep learning and computer vision techniques for image classification.

## Dataset

- **Source:** [South Indian Celebrity Face Dataset on Kaggle](https://www.kaggle.com/datasets/gunarakulangr/south-indian-celebrity-dataset)
- **Format:** One subfolder per celebrity, each containing facial images.

## How to Run

1. **Install required libraries:**
pip install torch torchvision matplotlib
2. **Prepare dataset:**  
Place all celebrity folders inside
south-indian-celebrity-dataset/
3. **Run:**  
Execute the main script:  
python python-code.py

## Model Architecture Choice

- **ResNet-18:**  
This model is chosen for its proven effectiveness in image recognition—with deep residual (skip) connections to address vanishing gradients, enabling stable training even with limited data.  
ResNet-18 strikes a strong balance between accuracy and resource use, making it ideal for real-world, mid-sized face recognition tasks.

## Preprocessing and Data Augmentation

- **Resizing:** Images are resized to (128x128) for consistent input and better computation.
- **Normalization:** Pixel values are normalized to [-1, 1], which accelerates and stabilizes learning.
- **Random Horizontal Flip:** Applied during training to boost data variability and improve the model's robustness to different facial orientations.

**Why:**  
These preprocessing steps help the model generalize better, especially when training data is not large or when variation in pose/lighting exists.

## Results

- Model training loss decreased consistently over 5 epochs.
- Training accuracy improved with each epoch, indicating effective pattern learning.
- **Test accuracy** reached ~0.17, which—while lower than training accuracy—demonstrates the model's basic generalization ability (see below for strengths/weaknesses).  
- Sample outputs: The model predicts the celebrity name with an associated confidence, e.g., “Predicted: surya, Confidence: 0.46”.
  <img width="504" height="230" alt="Screenshot 2025-10-29 203805" src="https://github.com/user-attachments/assets/dfff36af-4d29-45b8-a34c-449c6a4fcd8d" />
  <img width="1510" height="684" alt="Screenshot 2025-10-29 203702" src="https://github.com/user-attachments/assets/082df07b-77b0-4e2e-9c15-f5d89f0b1325" />
<img width="563" height="568" alt="Screenshot 2025-10-29 203926" src="https://github.com/user-attachments/assets/3730e4a3-e435-4a38-9206-b0c389941786" />

## Strengths

- Training accuracy and loss both improved promptly, confirming the network's ability to learn facial patterns in the dataset.
- Augmentation (flipping) helped slightly prevent overfitting and increase robustness versus training-only data.

## Weaknesses

- The test accuracy is modest, indicating possible overfitting and limited generalization to unseen faces.
- Causes may include limited training images per celebrity, lack of pose/background diversity, or high similarity between samples.

## Model Interpretation

- The model's learning curve demonstrates solid convergence on the provided data.
- Prominent gap between train and test accuracy highlights the importance of collecting more varied and balanced data for further improvement.

## Author
Kaviya M


