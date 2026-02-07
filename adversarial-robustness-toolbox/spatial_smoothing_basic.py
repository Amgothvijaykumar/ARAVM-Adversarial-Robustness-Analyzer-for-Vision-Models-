"""
SpatialSmoothing: Applies median filter to smooth adversarial perturbations
Paper: https://arxiv.org/abs/1704.01155

This defence uses a local spatial smoothing (median filter) to remove 
high-frequency adversarial noise while preserving image structure.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.defences.preprocessor import SpatialSmoothing
from art.utils import load_mnist

# Step 1: Load data and create classifier
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Prepare data for PyTorch
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Create simple CNN model
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

# Step 2: Create classifier WITHOUT defence first
classifier_no_defence = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Train the classifier
classifier_no_defence.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 3: Create SpatialSmoothing defence
spatial_smoothing = SpatialSmoothing(
    window_size=3,              # Size of median filter window (3x3)
    channels_first=True,        # PyTorch uses channels-first format
    clip_values=(min_pixel_value, max_pixel_value),
    apply_fit=False,            # Don't apply during training
    apply_predict=True          # Apply during prediction
)

print("SpatialSmoothing defence created:")
print(f"  Window size: {spatial_smoothing.window_size}")
print(f"  Channels first: {spatial_smoothing.channels_first}")

# Step 4: Create classifier WITH defence
classifier_with_defence = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    preprocessing_defences=[spatial_smoothing]  # Add defence here
)

# Step 5: Generate adversarial examples
attack = FastGradientMethod(estimator=classifier_no_defence, eps=0.3)
x_test_adv = attack.generate(x=x_test[:1000])

# Step 6: Evaluate both classifiers
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Clean accuracy (both should be similar)
predictions_clean = classifier_no_defence.predict(x_test[:1000])
accuracy_clean = np.sum(
    np.argmax(predictions_clean, axis=1) == np.argmax(y_test[:1000], axis=1)
) / 1000
print(f"\nClean Accuracy: {accuracy_clean * 100:.2f}%")

# Adversarial accuracy without defence
predictions_adv_no_def = classifier_no_defence.predict(x_test_adv)
accuracy_adv_no_def = np.sum(
    np.argmax(predictions_adv_no_def, axis=1) == np.argmax(y_test[:1000], axis=1)
) / 1000
print(f"\nWithout Defence:")
print(f"  Adversarial Accuracy: {accuracy_adv_no_def * 100:.2f}%")

# Adversarial accuracy with SpatialSmoothing defence
predictions_adv_with_def = classifier_with_defence.predict(x_test_adv)
accuracy_adv_with_def = np.sum(
    np.argmax(predictions_adv_with_def, axis=1) == np.argmax(y_test[:1000], axis=1)
) / 1000
print(f"\nWith SpatialSmoothing Defence:")
print(f"  Adversarial Accuracy: {accuracy_adv_with_def * 100:.2f}%")
print(f"  Improvement: {(accuracy_adv_with_def - accuracy_adv_no_def) * 100:.2f}%")