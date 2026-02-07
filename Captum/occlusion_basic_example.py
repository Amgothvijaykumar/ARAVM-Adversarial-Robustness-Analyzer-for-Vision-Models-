import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

from captum.attr import Occlusion
from captum.attr import visualization as viz

# 1. Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# 2. Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open('path/to/image.jpg')
input_tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 224, 224)

# 3. Get baseline prediction
with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()
    baseline_confidence = probabilities[0, predicted_class].item()

print(f'Predicted class: {predicted_class}')
print(f'Baseline confidence: {baseline_confidence:.4f}')

# 4. Initialize Occlusion
occlusion = Occlusion(model)

# 5. Define occlusion parameters
# For input shape (1, 3, 224, 224), sliding_window_shapes excludes batch dimension
# Format: (channels, height, width)

# Small patch (like a pixel-level occlusion)
small_patch = (3, 8, 8)  # Occlude 8x8 pixels across all channels

# Medium patch (standard patch attack)
medium_patch = (3, 16, 16)  # Occlude 16x16 pixels

# Large patch (more aggressive occlusion)
large_patch = (3, 32, 32)  # Occlude 32x32 pixels

# 6. Compute attributions with different patch sizes
attributions = occlusion.attribute(
    input_tensor,
    target=predicted_class,
    sliding_window_shapes=medium_patch,
    strides=(3, 8, 8),  # How much to shift the window (channels, height, width)
    baselines=0,  # Use 0 as baseline (black patches)
    perturbations_per_eval=16,  # Evaluate multiple occlusions in parallel
    show_progress=True
)

print(f'Attribution shape: {attributions.shape}')
print(f'Attribution range: [{attributions.min():.4f}, {attributions.max():.4f}]')