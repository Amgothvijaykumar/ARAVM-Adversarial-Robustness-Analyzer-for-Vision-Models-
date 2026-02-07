"""
Basic accuracy calculation before and after attack
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.utils import load_mnist

# Step 1: Load data and create classifier
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Prepare data for PyTorch (NCHW format)
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Create model and classifier
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv_1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv_2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = torch.nn.functional.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Train the classifier
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 2: Calculate accuracy on clean (benign) data
predictions_clean = classifier.predict(x_test)
accuracy_clean = np.sum(np.argmax(predictions_clean, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Accuracy on clean test examples: {accuracy_clean * 100:.2f}%")

# Step 3: Generate adversarial examples
attack = FastGradientMethod(estimator=classifier, eps=0.3)
x_test_adv = attack.generate(x=x_test)

# Step 4: Calculate accuracy on adversarial data
predictions_adv = classifier.predict(x_test_adv)
accuracy_adv = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Accuracy on adversarial test examples: {accuracy_adv * 100:.2f}%")

# Step 5: Calculate attack success rate
attack_success_rate = 1 - (accuracy_adv / accuracy_clean) if accuracy_clean > 0 else 0
print(f"Attack success rate: {attack_success_rate * 100:.2f}%")