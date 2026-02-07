"""
FastGradientMethod Parameter Guide:
"""

attack = FastGradientMethod(
    estimator=classifier,      # REQUIRED: Your trained ART classifier
    
    # Main attack parameters:
    eps=0.3,                   # Maximum perturbation (most important parameter!)
                              # - For images in [0,1]: try 0.01-0.3
                              # - For images in [0,255]: try 1-50
                              # - Higher eps = stronger attack but more visible
    
    norm=np.inf,              # Norm constraint for perturbation
                              # - np.inf: L-infinity norm (FGSM original)
                              # - 1: L1 norm
                              # - 2: L2 norm
    
    eps_step=0.1,             # Step size (only used with minimal=True)
    
    targeted=False,           # Attack type:
                              # - False: Untargeted (misclassify to any wrong class)
                              # - True: Targeted (misclassify to specific class)
    
    num_random_init=0,        # Number of random restarts
                              # - 0: Start from original image
                              # - >0: Multiple random starting points (stronger)
    
    batch_size=32,            # Batch size for processing
    
    minimal=False,            # Find minimal perturbation needed
                              # - True: Search for smallest eps that works
                              # - False: Use specified eps value
)

# Generating adversarial examples:
x_adv = attack.generate(x=x_test)                    # Untargeted attack
x_adv = attack.generate(x=x_test, y=y_test)          # Can provide labels
x_adv = attack.generate(x=x_test, y=target_labels)   # Targeted attack