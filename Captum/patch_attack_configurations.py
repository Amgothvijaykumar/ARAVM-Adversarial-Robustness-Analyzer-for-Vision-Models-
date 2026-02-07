"""
Understanding sliding_window_shapes and strides for different attack scenarios
"""

# For input tensor shape: (batch_size, channels, height, width)
# Example: (1, 3, 224, 224) for ImageNet models

# ============================================
# CONFIGURATION 1: Small Sliding Window (Fine-grained)
# ============================================
sliding_window_shapes = (3, 8, 8)  # 8x8 pixel patches
strides = (3, 4, 4)  # Slide by 4 pixels (50% overlap)

# Number of evaluations: ~(224/4) * (224/4) = ~3,136 patches
# Use case: Detailed attribution map, identify specific pixels

# ============================================
# CONFIGURATION 2: Medium Patch Attack
# ============================================
sliding_window_shapes = (3, 16, 16)  # 16x16 pixel patches
strides = (3, 8, 8)  # Slide by 8 pixels (50% overlap)

# Number of evaluations: ~(224/8) * (224/8) = ~784 patches
# Use case: Balance between detail and computation, standard patch attack

# ============================================
# CONFIGURATION 3: Large Patch Attack
# ============================================
sliding_window_shapes = (3, 32, 32)  # 32x32 pixel patches
strides = (3, 16, 16)  # Slide by 16 pixels (50% overlap)

# Number of evaluations: ~(224/16) * (224/16) = ~196 patches
# Use case: Fast computation, identify large important regions

# ============================================
# CONFIGURATION 4: Full Stride (No Overlap)
# ============================================
sliding_window_shapes = (3, 16, 16)
strides = (3, 16, 16)  # Same as window size = no overlap

# Number of evaluations: (224/16) * (224/16) = 196 patches
# Use case: Fastest, non-overlapping patches

# ============================================
# CONFIGURATION 5: Single Channel Occlusion
# ============================================
# To occlude only one channel at a time (e.g., test color importance)
sliding_window_shapes = (1, 224, 224)  # Full spatial, single channel
strides = (1, 224, 224)

# ============================================
# CONFIGURATION 6: Adversarial Patch Simulation
# ============================================
sliding_window_shapes = (3, 50, 50)  # Large adversarial patch
strides = (3, 25, 25)  # Significant overlap to test all positions

# Use case: Find where an adversarial patch would be most effective