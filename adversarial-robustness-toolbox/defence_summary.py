"""
Summary of preprocessing defences
"""

print("""
WHEN TO USE EACH DEFENCE:

SpatialSmoothing:
- Use when: Defending against gradient-based attacks (FGSM, PGD)
- Pros: Fast, effective against high-frequency noise
- Cons: May blur important features
- Parameters: window_size (3-7 recommended)

JPEG Compression:
- Use when: Defending against small perturbations
- Pros: Well-understood, removes compression artifacts
- Cons: Quality degradation, less effective at high quality
- Parameters: quality (50-75 recommended)

Combined Approach:
- Use when: Need strong defence against multiple attack types
- Pros: Multiple layers of protection
- Cons: Higher computational cost, larger accuracy drop
- Strategy: Feature Squeezing → JPEG → Spatial Smoothing

IMPORTANT: No defence is perfect. Always:
1. Test against adaptive attacks
2. Monitor clean accuracy
3. Use as part of a layered security approach
4. Consider adversarial training as an alternative
""")