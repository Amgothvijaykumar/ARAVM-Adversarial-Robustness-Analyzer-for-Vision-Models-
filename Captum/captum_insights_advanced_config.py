"""
Advanced configuration options for Captum Insights
"""

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature
import torch.nn.functional as F

# ============================================
# Multiple Models Comparison
# ============================================

# Compare different models
model_resnet50 = models.resnet50(pretrained=True).eval()
model_resnet34 = models.resnet34(pretrained=True).eval()

visualizer = AttributionVisualizer(
    models=[model_resnet50, model_resnet34],  # Compare multiple models
    score_func=lambda o: F.softmax(o, 1),
    classes=get_imagenet_classes(),
    features=[
        ImageFeature(
            "Image",
            baseline_transforms=[baseline_func],
            input_transforms=[normalize],
        )
    ],
    dataset=formatted_data_iter(),
)

# ============================================
# Custom Baseline Transforms
# ============================================

# Use multiple baseline strategies
visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: F.softmax(o, 1),
    classes=get_imagenet_classes(),
    features=[
        ImageFeature(
            "Image",
            baseline_transforms=[
                lambda x: x * 0,  # Black baseline
                lambda x: torch.ones_like(x) * 0.5,  # Gray baseline
            ],
            input_transforms=[normalize],
        )
    ],
    dataset=formatted_data_iter(),
)

# ============================================
# Custom Visualization Transform
# ============================================

def unnormalize_for_display(img_tensor):
    """
    Unnormalize image for visualization
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    return torch.clamp(img, 0, 1)

visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: F.softmax(o, 1),
    classes=get_imagenet_classes(),
    features=[
        ImageFeature(
            "Image",
            baseline_transforms=[baseline_func],
            input_transforms=[normalize],
            visualization_transform=unnormalize_for_display,  # Custom display
        )
    ],
    dataset=formatted_data_iter(),
)

# ============================================
# For Binary Classification
# ============================================

# When model outputs single value (not per-class)
binary_model = YourBinaryModel()

visualizer = AttributionVisualizer(
    models=[binary_model],
    score_func=lambda o: torch.sigmoid(o),  # For binary output
    classes=["Negative", "Positive"],
    features=[ImageFeature("Image", ...)],
    dataset=your_data_iter(),
    use_label_for_attr=False,  # Important for binary models!
)