"""
Launch Captum Insights as a standalone server
(Not embedded in notebook)
"""

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature
from torchvision import models, transforms
import torch.nn.functional as F

# Setup (same as before)
model = models.resnet50(pretrained=True).eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: F.softmax(o, 1),
    classes=get_imagenet_classes(),
    features=[
        ImageFeature(
            "Image",
            baseline_transforms=[lambda x: x * 0],
            input_transforms=[normalize],
        )
    ],
    dataset=formatted_data_iter(),
)

# Launch as a server (accessible via browser)
visualizer.serve(
    debug=True,
    port=8080,  # Custom port
    blocking=False,  # Non-blocking mode
    bind_all=True  # Allow external connections
)

# This will print a URL like: http://localhost:8080
# Navigate to it in your browser