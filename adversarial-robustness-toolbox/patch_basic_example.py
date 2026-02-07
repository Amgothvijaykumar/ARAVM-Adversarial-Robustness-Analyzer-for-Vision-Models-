"""
Basic example: Create and train an adversarial patch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from art.attacks.evasion.adversarial_patch import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier

# Step 1: Load or create a classifier
model = models.resnet18(pretrained=True)
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=1000,
    clip_values=(0.0, 1.0),
    channels_first=True,
)

# Step 2: Initialize AdversarialPatchPyTorch
attack = AdversarialPatchPyTorch(
    estimator=classifier,
    rotation_max=22.5,        # Maximum rotation in degrees (±22.5°)
    scale_min=0.1,            # Minimum patch scale (10% of image)
    scale_max=1.0,            # Maximum patch scale (100% of image)
    learning_rate=5.0,        # Learning rate for optimization
    max_iter=500,             # Number of optimization iterations
    batch_size=16,            # Batch size for training
    patch_shape=(3, 224, 224),  # Patch shape (CHW): 224x224 RGB patch
    patch_type='circle',      # 'circle' or 'square'
    optimizer='Adam',         # 'Adam' or 'pgd'
    targeted=True,            # Targeted attack (specify target class)
    verbose=True              # Show progress bar
)

# Step 3: Prepare training data
# Load some images (e.g., ImageNet samples)
x_train = np.random.rand(100, 3, 224, 224).astype(np.float32)  # Example data
y_target = np.zeros((100, 1000))
y_target[:, 500] = 1  # Target all images to be classified as class 500

# Step 4: Generate the adversarial patch
print("Generating adversarial patch...")
patch, patch_mask = attack.generate(x=x_train, y=y_target)

print(f"Patch shape: {patch.shape}")
print(f"Patch mask shape: {patch_mask.shape}")

# The patch is now trained and can be applied to new images!