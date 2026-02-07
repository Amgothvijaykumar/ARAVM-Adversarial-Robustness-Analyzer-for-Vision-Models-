import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from PIL import Image
from captum.attr import Occlusion
import matplotlib.pyplot as plt

class OcclusionAnalyzer:
    """Analyze model behavior under occlusion attacks"""
    
    def __init__(self, model, input_tensor, target_class=None):
        self.model = model
        self.model.eval()
        self.input_tensor = input_tensor
        
        # Get baseline prediction
        with torch.no_grad():
            self.baseline_output = model(input_tensor)
            self.baseline_probs = F.softmax(self.baseline_output, dim=1)
            
        if target_class is None:
            self.target_class = self.baseline_output.argmax(dim=1).item()
        else:
            self.target_class = target_class
            
        self.baseline_confidence = self.baseline_probs[0, self.target_class].item()
        
    def compute_occlusion_attributions(self, 
                                      sliding_window_shapes,
                                      strides=None,
                                      baselines=0):
        """Compute occlusion attributions"""
        occlusion = Occlusion(self.model)
        
        if strides is None:
            # Default: stride = half of window size
            strides = tuple(s // 2 for s in sliding_window_shapes)
        
        attributions = occlusion.attribute(
            self.input_tensor,
            target=self.target_class,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines,
            perturbations_per_eval=32,  # Batch multiple occlusions
            show_progress=True
        )
        
        return attributions
    
    def test_specific_occlusion(self, mask):
        """
        Test occlusion at a specific location with a custom mask
        
        Args:
            mask: Binary mask (1 = keep, 0 = occlude), same shape as input
        """
        occluded_input = self.input_tensor * mask
        
        with torch.no_grad():
            output = self.model(occluded_input)
            probs = F.softmax(output, dim=1)
            
        confidence_after = probs[0, self.target_class].item()
        confidence_change = confidence_after - self.baseline_confidence
        
        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
        
        results = {
            'confidence_before': self.baseline_confidence,
            'confidence_after': confidence_after,
            'confidence_change': confidence_change,
            'confidence_change_percent': (confidence_change / self.baseline_confidence) * 100,
            'top5_classes': top5_indices[0].cpu().numpy(),
            'top5_probs': top5_probs[0].cpu().numpy(),
            'prediction_changed': top5_indices[0, 0].item() != self.target_class
        }
        
        return results
    
    def find_most_important_patch(self, attributions, patch_size):
        """
        Find the patch location that causes maximum attribution
        
        Args:
            attributions: Attribution tensor
            patch_size: (height, width) of patch to find
            
        Returns:
            Location and importance of most critical patch
        """
        # Average across channels if needed
        if len(attributions.shape) == 4:  # (batch, channels, h, w)
            attr_map = attributions[0].mean(dim=0)  # (h, w)
        else:
            attr_map = attributions[0]
            
        # Find location with maximum absolute attribution
        abs_attr = torch.abs(attr_map)
        
        # Use sliding window to find best patch location
        h, w = attr_map.shape
        ph, pw = patch_size
        
        max_importance = -float('inf')
        best_location = None
        
        for i in range(h - ph + 1):
            for j in range(w - pw + 1):
                patch_importance = abs_attr[i:i+ph, j:j+pw].sum().item()
                if patch_importance > max_importance:
                    max_importance = patch_importance
                    best_location = (i, j)
        
        return best_location, max_importance
    
    def measure_patch_attack_effectiveness(self, 
                                          sliding_window_shapes,
                                          num_positions=10):
        """
        Measure how effective patches are at different positions
        """
        attributions = self.compute_occlusion_attributions(sliding_window_shapes)
        
        # Create mask for occluding a patch
        c, h, w = sliding_window_shapes
        img_h, img_w = self.input_tensor.shape[2], self.input_tensor.shape[3]
        
        results = []
        
        # Test patches at different positions
        positions = []
        step_h = img_h // int(np.sqrt(num_positions))
        step_w = img_w // int(np.sqrt(num_positions))
        
        for i in range(0, img_h - h, step_h):
            for j in range(0, img_w - w, step_w):
                positions.append((i, j))
        
        for i, j in positions[:num_positions]:
            # Create mask
            mask = torch.ones_like(self.input_tensor)
            mask[:, :, i:i+h, j:j+w] = 0  # Occlude this region
            
            result = self.test_specific_occlusion(mask)
            result['position'] = (i, j)
            result['attribution_sum'] = attributions[:, :, i:i+h, j:j+w].sum().item()
            results.append(result)
        
        return results


# ============================================
# USAGE EXAMPLE
# ============================================

# Load model and image
model = models.resnet50(pretrained=True)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open('path/to/image.jpg')
input_tensor = transform(img).unsqueeze(0)

# Initialize analyzer
analyzer = OcclusionAnalyzer(model, input_tensor)

print(f"Target class: {analyzer.target_class}")
print(f"Baseline confidence: {analyzer.baseline_confidence:.4f}")

# Test different patch sizes
patch_configs = [
    (3, 16, 16, "Small"),
    (3, 32, 32, "Medium"),
    (3, 64, 64, "Large")
]

for *patch_size, name in patch_configs:
    print(f"\n{'='*50}")
    print(f"Testing {name} patch: {patch_size}")
    print(f"{'='*50}")
    
    # Compute attributions
    attributions = analyzer.compute_occlusion_attributions(
        sliding_window_shapes=tuple(patch_size),
        strides=tuple(s // 2 for s in patch_size)
    )
    
    # Find most important patch
    location, importance = analyzer.find_most_important_patch(
        attributions, 
        patch_size=(patch_size[1], patch_size[2])
    )
    
    print(f"Most critical patch location: {location}")
    print(f"Importance score: {importance:.4f}")
    
    # Test actual occlusion at that location
    mask = torch.ones_like(input_tensor)
    i, j = location
    mask[:, :, i:i+patch_size[1], j:j+patch_size[2]] = 0
    
    result = analyzer.test_specific_occlusion(mask)
    print(f"Confidence drop: {result['confidence_change']:.4f} "
          f"({result['confidence_change_percent']:.1f}%)")
    print(f"Prediction changed: {result['prediction_changed']}")
    
# Comprehensive position analysis
print(f"\n{'='*50}")
print("Testing multiple patch positions")
print(f"{'='*50}")

results = analyzer.measure_patch_attack_effectiveness(
    sliding_window_shapes=(3, 32, 32),
    num_positions=9
)

# Sort by confidence change
results_sorted = sorted(results, key=lambda x: x['confidence_change'])

print("\nTop 5 most damaging patch positions:")
for i, r in enumerate(results_sorted[:5], 1):
    print(f"{i}. Position {r['position']}: "
          f"Confidence {r['confidence_change']:.4f} "
          f"({r['confidence_change_percent']:.1f}% change)")