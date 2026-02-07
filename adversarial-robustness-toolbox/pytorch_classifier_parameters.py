"""
PyTorchClassifier Required Parameters:
"""

classifier = PyTorchClassifier(
    # REQUIRED PARAMETERS:
    model=model,              # torch.nn.Module - Your PyTorch model (returns logits preferred)
    loss=criterion,           # torch.nn.modules.loss._Loss - Loss function (e.g., CrossEntropyLoss)
    input_shape=(3, 224, 224),  # tuple - Shape of one input sample (C, H, W for channels_first=True)
    nb_classes=10,            # int - Number of output classes
    
    # OPTIONAL BUT RECOMMENDED:
    optimizer=optimizer,      # torch.optim.Optimizer - Required if you plan to train
    clip_values=(0.0, 1.0),  # tuple - Min/max values for input normalization
    channels_first=True,      # bool - True for PyTorch (NCHW format)
    
    # OPTIONAL ADVANCED:
    preprocessing=(mean, std),  # tuple - Preprocessing normalization
    use_amp=False,            # bool - Use automatic mixed precision
    device_type="gpu",        # str - "gpu" or "cpu"
    preprocessing_defences=None,  # Defence mechanisms
    postprocessing_defences=None,
)