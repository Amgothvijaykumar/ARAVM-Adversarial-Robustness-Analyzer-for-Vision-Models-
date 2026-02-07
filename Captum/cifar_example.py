from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import torch
import numpy as np

# Assuming you have a trained model and test image
# model = YourCIFARModel()
# test_image = ...  # Shape: (1, 3, 32, 32)

# Initialize Integrated Gradients
ig = IntegratedGradients(model)

# Calculate attributions
attr_ig, delta = ig.attribute(
    test_image, 
    target=predicted_class,
    n_steps=50,
    return_convergence_delta=True
)

print(f'Approximation delta: {delta}')

# Convert for visualization
attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
original_image = np.transpose((test_image.squeeze().cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

# Visualize side-by-side
_ = viz.visualize_image_attr(
    attr_ig,
    original_image,
    method="blended_heat_map",
    sign="all",
    show_colorbar=True,
    title="Overlayed Integrated Gradients"
)