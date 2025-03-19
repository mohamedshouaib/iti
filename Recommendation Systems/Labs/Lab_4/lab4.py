# PyTorch Tutorial: Comprehensive Guide

## Introduction to PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available and set device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## 1. Tensors - The Fundamental Data Structure in PyTorch

# Creating tensors
print("Creating different types of tensors:")

# From Python lists
x = torch.tensor([1, 2, 3, 4])
print(f"Tensor from list: {x}")

# From NumPy arrays
np_array = np.array([1, 2, 3, 4])
x_from_numpy = torch.from_numpy(np_array)
print(f"Tensor from NumPy: {x_from_numpy}")

# Tensor with specific data type
x_float = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print(f"Float tensor: {x_float}")

# Creating tensors with specific shapes
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
rand_tensor = torch.rand(3, 3)

print("\nTensors with specific shapes:")
print(f"Zeros tensor (3x4):\n{zeros}")
print(f"Ones tensor (2x3):\n{ones}")
print(f"Random tensor (3x3):\n{rand_tensor}")

# Tensor operations
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

print("\nTensor operations:")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"Matrix multiplication: {torch.matmul(a, b)}")

# Reshaping tensors
c = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\nOriginal tensor:\n{c}")
print(f"Reshaped tensor:\n{c.reshape(3, 2)}")
print(f"Transposed tensor:\n{c.t()}")

# Moving tensors to GPU if available
if torch.cuda.is_available():
    x_gpu = x.to(device)
    print(f"\nTensor moved to GPU: {x_gpu.device}")

## 2. Autograd - Automatic Differentiation

# Creating tensors with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3*x + 1

print("\nAutograd example:")
print(f"y = x^2 + 3x + 1 = {y} when x = {x}")

# Computing gradients
y.backward()
print(f"dy/dx = 2x + 3 = {x.grad} when x = {x}")

# Example of gradient accumulation and zeroing
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(f"First gradient: {x.grad}")

# Gradient will accumulate if we don't zero it
y = x ** 3
y.backward()
print(f"Accumulated gradient (should be wrong): {x.grad}")

# Zero the gradient and compute again
x.grad.zero_()
y = x ** 3
y.backward()
print(f"Correct gradient after zeroing: {x.grad}")

## 3. Neural Networks with nn.Module

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create a model instance
model = SimpleNN(input_size=10, hidden_size=20, output_size=2)
print("\nSimple Neural Network:")
print(model)

# Random input
x = torch.randn(5, 10)  # 5 samples, 10 features each
print(f"\nInput shape: {x.shape}")

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")

# Access model parameters
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

## 4. Loss Functions and Optimizers

# Create some dummy data
inputs = torch.randn(100, 10)
targets = torch.randint(0, 2, (100,))  # Binary classification

# Create a model
model = SimpleNN(input_size=10, hidden_size=20, output_size=2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("\nTraining example:")
# Train for 5 epochs
for epoch in range(5):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

## 5. Datasets and DataLoaders

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

# Create DataLoaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

print("\nDataset and DataLoader:")
print(f"Number of training samples: {len(trainset)}")
print(f"Number of test samples: {len(testset)}")
print(f"Number of batches in trainloader: {len(trainloader)}")

# Visualize a batch of data
dataiter = iter(trainloader)
images, labels = next(dataiter)

print(f"Batch shape: {images.shape}")
print(f"Labels: {labels[:8]}")

# Function to display images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# Display a batch of images
print("\nSample images from MNIST dataset:")
# Uncomment the following line to display images when running in a notebook
# imshow(torchvision.utils.make_grid(images[:8]))

## 6. Convolutional Neural Networks (CNNs)

# Define a CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a CNN model
cnn_model = SimpleCNN()
print("\nConvolutional Neural Network:")
print(cnn_model)

## 7. Training a CNN on MNIST

# Move model to device (GPU if available)
cnn_model = cnn_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Train function
def train_model(model, loader, criterion, optimizer, num_epochs=2):
    print("\nTraining CNN on MNIST:")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print statistics every 100 mini-batches
            if i % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}, "
                      f"Accuracy: {100 * correct / total:.2f}%")
                running_loss = 0.0
        
        print(f"Epoch {epoch+1} completed")

# Comment out the following line to skip training
# train_model(cnn_model, trainloader, criterion, optimizer, num_epochs=2)

## 8. Evaluating the Model

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# Comment out the following line to skip evaluation
# evaluate_model(cnn_model, testloader)

## 9. Saving and Loading Models

# Save model
model_path = "mnist_cnn.pth"
# torch.save(cnn_model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")

# Load model
new_model = SimpleCNN()
# new_model.load_state_dict(torch.load(model_path))
# new_model.to(device)
print("Model loaded successfully")

## 10. Transfer Learning

# Define a new model based on a pre-trained network
class TransferModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TransferModel, self).__init__()
        # Use a pre-trained ResNet18 model
        self.resnet = torchvision.models.resnet18(pretrained=True)
        
        # Replace the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Modify the first convolution layer to accept grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        return self.resnet(x)

# Create transfer learning model
transfer_model = TransferModel()
print("\nTransfer Learning Model:")
print(f"Using ResNet18 with modified input layer and output layer")

## 11. Advanced Techniques: Learning Rate Scheduler

# Define a learning rate scheduler
optimizer = optim.SGD(transfer_model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print("\nLearning rate scheduler:")
print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

# Simulate training for 15 epochs
for epoch in range(15):
    # Training code would go here
    
    # Step the scheduler
    scheduler.step()
    
    if epoch % 3 == 0:
        print(f"Epoch {epoch}: Learning rate = {optimizer.param_groups[0]['lr']}")

## 12. Custom Datasets

# Define a custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]

# Create a simple custom dataset
data = torch.randn(100, 1, 28, 28)  # 100 grayscale images of size 28x28
labels = torch.randint(0, 10, (100,))  # Random labels between 0 and 9

custom_dataset = CustomDataset(data, labels)
custom_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=10, shuffle=True)

print("\nCustom Dataset:")
print(f"Dataset size: {len(custom_dataset)}")
print(f"Batch size: {next(iter(custom_loader))[0].shape}")

## 13. Deployment: Converting to TorchScript

# Convert model to TorchScript
scripted_model = torch.jit.script(cnn_model)

# Save scripted model
scripted_model_path = "mnist_cnn_scripted.pt"
# scripted_model.save(scripted_model_path)
print(f"\nTorchScript model saved to {scripted_model_path}")

## 14. Project: Putting it All Together

print("\n--- Final Project: MNIST Classifier ---")

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False,
