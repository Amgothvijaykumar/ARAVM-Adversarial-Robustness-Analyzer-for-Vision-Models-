"""
Complete parameter guide for AttributionVisualizer
"""

AttributionVisualizer(
    # REQUIRED PARAMETERS
    models=[model1, model2],  # List of PyTorch models to visualize
    classes=["class1", "class2", ...],  # List of class names
    features=[ImageFeature(...)],  # Feature definitions
    dataset=data_iterator(),  # Iterator yielding Batch objects
    
    # OPTIONAL PARAMETERS
    score_func=lambda o: F.softmax(o, 1),
    # Function to convert model output to class scores
    # - For classification: use softmax
    # - For binary: use sigmoid
    # - If None: uses raw model output
    
    use_label_for_attr=True,
    # Whether to pass class index to attribution method
    # - True: For multi-class (most cases)
    # - False: For binary classification with single output
)

# ImageFeature parameters
ImageFeature(
    name="Image",  # Display name for this feature
    
    baseline_transforms=[func1, func2],
    # List of functions to create baselines
    # Each function takes input tensor and returns baseline
    # Common baselines:
    # - lambda x: x * 0  (black image)
    # - lambda x: x.mean()  (mean color)
    # - lambda x: blur(x)  (blurred version)
    
    input_transforms=[normalize],
    # Transforms applied before passing to model
    # Typically normalization
    
    visualization_transform=unnormalize_func,
    # Optional: Transform for display only
    # Use to unnormalize for better visualization
)

# Batch object structure
Batch(
    inputs=torch.Tensor(...),  # Shape: (batch_size, channels, height, width)
    labels=torch.Tensor(...),  # Shape: (batch_size,) - class indices
    additional_args=None,  # Optional: Additional model inputs
)