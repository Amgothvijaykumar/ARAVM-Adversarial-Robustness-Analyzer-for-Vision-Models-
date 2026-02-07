"""
Common issues and solutions for Captum Insights
"""

# Issue 1: Widget not displaying
# Solution: Make sure ipywidgets is installed and enabled
# !pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension

# Issue 2: Images look weird (too bright/dark)
# Solution: Make sure images are in [0, 1] range before normalization
def check_image_range(img_tensor):
    print(f"Image range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    if img_tensor.max() > 1.0:
        print("WARNING: Image values > 1.0, consider dividing by 255")
    return img_tensor

# Issue 3: Model expects normalized input but visualizer shows unnormalized
# Solution: Use visualization_transform parameter
def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

# Issue 4: "Iterator exhausted" error
# Solution: Make sure your data iterator is infinite or has enough samples
def infinite_data_iter():
    while True:  # Infinite loop
        for batch in your_dataloader:
            yield Batch(inputs=batch[0], labels=batch[1])

# Issue 5: Slow attribution computation
# Solution: Reduce n_perturb_samples or use faster methods
# In the widget interface, you can adjust:
# - n_steps (for Integrated Gradients)
# - n_samples (for SHAP methods)

# Issue 6: Memory errors with large images
# Solution: Reduce batch size or image resolution
transform = transforms.Compose([
    transforms.Resize(224),  # Smaller size
    transforms.CenterCrop(224),
    transforms.ToTensor()
])