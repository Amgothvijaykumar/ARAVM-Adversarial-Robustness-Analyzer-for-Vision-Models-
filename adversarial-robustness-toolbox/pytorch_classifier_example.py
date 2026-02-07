import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from art.estimators.classification import PyTorchClassifier

# Step 1: Load a pre-trained torchvision model
model = models.resnet18(pretrained=True)

# Optional: Modify the final layer for your number of classes (if different from ImageNet's 1000)
# For example, for CIFAR-10 with 10 classes:
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Step 2: Define the loss function
criterion = nn.CrossEntropyLoss()

# Step 3: Define the optimizer
# Common optimizers for pre-trained models:

# Option A: SGD with momentum (typical for fine-tuning)
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4
)

# Option B: Adam optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Create the PyTorchClassifier wrapper
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 224, 224),  # torchvision models typically expect 224x224
    nb_classes=num_classes,
    clip_values=(0.0, 1.0),  # Pixel value range
    channels_first=True,  # PyTorch uses NCHW format
)

# Optional: Add preprocessing (ImageNet normalization)
# classifier = PyTorchClassifier(
#     model=model,
#     loss=criterion,
#     optimizer=optimizer,
#     input_shape=(3, 224, 224),
#     nb_classes=num_classes,
#     clip_values=(0.0, 1.0),
#     channels_first=True,
#     preprocessing=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std
# )