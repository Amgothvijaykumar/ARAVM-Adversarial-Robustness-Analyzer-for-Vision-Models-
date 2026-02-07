import matplotlib.pyplot as plt
import numpy as np
from captum.attr import visualization as viz

def visualize_occlusion_results(input_tensor, attributions, 
                                original_image=None, num_cols=3):
    """
    Visualize occlusion attributions with multiple methods
    """
    # Prepare attribution data
    attr_data = attributions.squeeze().cpu().detach().numpy()
    if len(attr_data.shape) == 3:  # (C, H, W)
        attr_data = np.transpose(attr_data, (1, 2, 0))  # (H, W, C)
    
    # Prepare original image if not provided
    if original_image is None:
        original_image = input_tensor.squeeze().cpu().detach().numpy()
        if len(original_image.shape) == 3:
            original_image = np.transpose(original_image, (1, 2, 0))
        # Unnormalize (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_image = (original_image * std + mean)
        original_image = np.clip(original_image, 0, 1)
    
    # Create visualization grid
    methods = [
        ('original_image', 'Original Image', 'all'),
        ('heat_map', 'Attribution Heat Map', 'absolute_value'),
        ('blended_heat_map', 'Blended Attribution', 'all'),
    ]
    
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
    
    for idx, (method, title, sign) in enumerate(methods):
        if method == 'original_image':
            viz.visualize_image_attr(
                None,
                original_image,
                method=method,
                title=title,
                plt_fig_axis=(fig, axes[idx]),
                use_pyplot=False
            )
        else:
            viz.visualize_image_attr(
                attr_data,
                original_image,
                method=method,
                sign=sign,
                show_colorbar=True,
                title=title,
                plt_fig_axis=(fig, axes[idx]),
                use_pyplot=False
            )
    
    plt.tight_layout()
    return fig


def plot_confidence_vs_patch_position(results):
    """
    Plot how confidence changes at different patch positions
    """
    positions = [r['position'] for r in results]
    confidences = [r['confidence_after'] for r in results]
    baseline = results[0]['confidence_before']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Confidence after occlusion
    x_pos = range(len(positions))
    ax1.bar(x_pos, confidences, alpha=0.7, label='After Occlusion')
    ax1.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
    ax1.set_xlabel('Patch Position Index')
    ax1.set_ylabel('Confidence')
    ax1.set_title('Model Confidence After Occlusion at Different Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence change percentage
    changes = [r['confidence_change_percent'] for r in results]
    colors = ['red' if c < 0 else 'green' for c in changes]
    ax2.bar(x_pos, changes, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Patch Position Index')
    ax2.set_ylabel('Confidence Change (%)')
    ax2.set_title('Percentage Change in Confidence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_heatmap_overlay(input_tensor, attributions, alpha=0.6):
    """
    Create a heatmap overlay showing occlusion importance
    """
    # Get attribution map
    attr_map = attributions.squeeze().cpu().detach().numpy()
    if len(attr_map.shape) == 3:
        attr_map = np.mean(np.abs(attr_map), axis=0)  # Average across channels
    
    # Normalize to [0, 1]
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
    
    # Get original image
    img = input_tensor.squeeze().cpu().detach().numpy()
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean)
    img = np.clip(img, 0, 1)
    
    # Create heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.imshow(attr_map, cmap='hot', alpha=alpha)
    plt.colorbar(label='Attribution Importance')
    plt.title('Occlusion Importance Heatmap')
    plt.axis('off')
    
    return plt.gcf()