import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog

DATA_DIR = r"D:\OneDrive\Documents\Face Recognition Model\south-indian-celebrity-dataset"
IMG_SIZE = 128
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
class_names = dataset.classes
num_classes = len(class_names)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
test_set.dataset.transform = test_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
train_accuracies = []

def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f} Accuracy={epoch_acc:.4f}")

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    train()
    test()

    epochs = np.arange(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, marker='o')
    plt.title("Training Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "face_recognition_model.pth")
    print("Model saved as face_recognition_model.pth")

    print("Select an image for recognition...")
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=DATA_DIR, title="Select Test Image",
                                           filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")))
    root.destroy()

    if file_path:
        def predict_and_show(image_path):
            model.eval()
            img = Image.open(image_path).convert('RGB')
            img_tensor = test_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.softmax(output, dim=1)
                pred_class = prob.argmax(dim=1).item()
                confidence = prob[0, pred_class].item()
            predicted_name = class_names[pred_class]
            print(f"Predicted: {predicted_name}, Confidence: {confidence:.2f}")

            plt.figure()
            plt.imshow(np.array(img))
            plt.title(f"Predicted: {predicted_name}\nConfidence: {confidence:.2f}")
            plt.axis('off')
            plt.show()

        predict_and_show(file_path)
    else:
        print("No file selected.")
