"""
Understanding patch initialization
"""

# The patch is initialized automatically in __init__:
# Lines 175-179 of the source code:

# Calculate mean value between clip_values
mean_value = (estimator.clip_values[1] - estimator.clip_values[0]) / 2.0 + estimator.clip_values[0]

# Initialize patch as a tensor with all pixels set to mean_value
# For [0, 1] range: mean_value = 0.5 (gray)
# For [0, 255] range: mean_value = 127.5 (gray)
import torch
initial_value = np.ones(patch_shape) * mean_value
patch_tensor = torch.tensor(initial_value, requires_grad=True, device=estimator.device)

# You can reset the patch to different values:
attack.reset_patch(initial_patch_value=None)  # Reset to original mean value
attack.reset_patch(initial_patch_value=0.8)   # Reset to specific value
attack.reset_patch(initial_patch_value=custom_patch_array)  # Use custom patch