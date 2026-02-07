import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# 1. Load and prepare your model
model = models.resnet18(pretrained=True)
model.eval()

# 2. Load and preprocess your image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load your image
img = Image.open('path/to/your/image.jpg')
input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# 3. Make prediction
output = model(input_tensor)
predicted_class = output.argmax(dim=1).item()
print(f'Predicted class: {predicted_class}')

# 4. Initialize Integrated Gradients
ig = IntegratedGradients(model)

# 5. Calculate attributions
# You can specify a target class, or use the predicted class
attributions, delta = ig.attribute(
    input_tensor,
    target=predicted_class,
    n_steps=50,  # Number of steps for the approximation
    return_convergence_delta=True  # Optional: returns approximation error
)

print(f'Approximation delta: {delta}')

# 6. Prepare attribution data for visualization
# Convert to numpy and transpose to (H, W, C) format
attr_data = attributions.squeeze().cpu().detach().numpy()
attr_data = np.transpose(attr_data, (1, 2, 0))

# Prepare original image for visualization (unnormalize and convert to numpy)
original_image = input_tensor.squeeze().cpu().detach().numpy()
original_image = np.transpose(original_image, (1, 2, 0))
# Unnormalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
original_image = (original_image * std + mean)
original_image = np.clip(original_image, 0, 1)

# 7. Visualize attributions side-by-side with original image

# Method 1: Blended heat map (overlays attribution on grayscale image)
fig, ax = viz.visualize_image_attr(
    attr_data,
    original_image,
    method='blended_heat_map',
    sign='all',  # Show both positive and negative attributions
    show_colorbar=True,
    title='Integrated Gradients - Blended Heat Map'
)

# Method 2: Heat map only
fig, ax = viz.visualize_image_attr(
    attr_data,
    original_image,
    method='heat_map',
    sign='absolute_value',  # Show absolute values
    show_colorbar=True,
    title='Integrated Gradients - Heat Map'
)

# Method 3: Multiple visualizations in subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
viz.visualize_image_attr(
    None,
    original_image,
    method='original_image',
    title='Original Image',
    plt_fig_axis=(fig, axes[0]),
    use_pyplot=False
)

# Overlayed attribution
viz.visualize_image_attr(
    attr_data,
    original_image,
    method='blended_heat_map',
    sign='all',
    show_colorbar=True,
    title='Overlayed Integrated Gradients',
    plt_fig_axis=(fig, axes[1]),
    use_pyplot=False
)

# Attribution only
viz.visualize_image_attr(
    attr_data,
    original_image,
    method='heat_map',
    sign='absolute_value',
    show_colorbar=True,
    title='Attribution Heat Map',
    plt_fig_axis=(fig, axes[2]),
    use_pyplot=False
)

plt.tight_layout()
plt.show()