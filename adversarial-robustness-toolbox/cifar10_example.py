import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from art.estimators.classification import PyTorchClassifier

# Define the model (could be from torchvision)
model = PreActResNet18()

# Define optimizer with learning rate scheduler
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)
lr_scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# Define loss
criterion = nn.CrossEntropyLoss()

# Define preprocessing (CIFAR-10 normalization)
cifar_mu = np.ones((3, 32, 32))
cifar_mu[0, :, :] = 0.4914
cifar_mu[1, :, :] = 0.4822
cifar_mu[2, :, :] = 0.4465

cifar_std = np.ones((3, 32, 32))
cifar_std[0, :, :] = 0.2471
cifar_std[1, :, :] = 0.2435
cifar_std[2, :, :] = 0.2616

# Create ART classifier with preprocessing
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),
    preprocessing=(cifar_mu, cifar_std),  # (mean, std) for normalization
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

# You can now use the classifier with fit, predict, etc.
# classifier.fit(x_train, y_train, batch_size=128, nb_epochs=10, scheduler=lr_scheduler)