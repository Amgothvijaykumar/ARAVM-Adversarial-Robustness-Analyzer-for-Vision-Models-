"""
Run ARAVM with a custom image (e.g., panda.png)
Usage: python run_with_image.py datasets/panda.png

Generates:
- aravm_dashboard.png (visual comparison)
- level1_heatmap_comparison.png (gradient heatmaps)
- security_report.html (comprehensive HTML report)
"""

import sys
import os
import torch
from PIL import Image
from main_analyzer import (
    AdversarialRobustnessAnalyzer,
    AttackConfig,
    DefenseConfig
)
from report_generator import generate_html_report

def main():
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "datasets/cat.jpg"
    
    image_name = os.path.basename(image_path)
    print(f"\n🐼 Running ARAVM on: {image_path}\n")
    
    # Initialize analyzer
    analyzer = AdversarialRobustnessAnalyzer(
        model_name="resnet50",
        attack_config=AttackConfig(
            epsilon=0.03,
            patch_width=50,
            patch_height=50
        ),
        defense_config=DefenseConfig(
            jpeg_quality=50
        )
    )
    
    # Load the image
    img = Image.open(image_path).convert('RGB')
    x = analyzer.victim.transform(img).numpy()
    
    # Get prediction
    pred, conf = analyzer.victim.predict(torch.from_numpy(x))
    
    print(f"\n📸 Image loaded successfully!")
    print(f"   Shape: {x.shape}")
    print(f"   Predicted Class: {pred}")
    print(f"   Confidence: {conf:.2%}")
    
    # Run noise slider analysis and capture results
    print("\n" + "="*60)
    epsilon_values = [0.01, 0.03, 0.05, 0.1, 0.2]
    slider_results = {}
    
    import numpy as np
    for eps in epsilon_values:
        x_adv = analyzer.level1.fgsm_attack(x[np.newaxis, ...], epsilon=eps)
        adv_pred, adv_conf = analyzer.victim.predict(torch.from_numpy(x_adv[0]))
        l2 = np.sqrt(np.sum((x_adv[0] - x) ** 2))
        fooled = adv_pred != pred
        
        slider_results[eps] = {
            'prediction': adv_pred,
            'confidence': adv_conf,
            'l2_distortion': l2,
            'success': fooled
        }
        
        status = "✗ FOOLED" if fooled else "✓ Robust"
        print(f"  ε={eps:.2f}: Pred={adv_pred} ({adv_conf:.2%}) | L2={l2:.3f} | {status}")
    
    print("-" * 60)
    
    # Run full audit (this generates the visualizations)
    reports = analyzer.run_full_audit(
        x, 
        y_true=pred,
        save_visualizations=True,
        output_dir="."
    )
    
    # Generate HTML Report
    print("\n" + "="*60)
    print("  GENERATING HTML SECURITY REPORT")
    print("="*60)
    
    # Prepare metrics for report
    x_batch = x[np.newaxis, ...]
    
    # FGSM metrics
    x_fgsm = analyzer.level1.fgsm_attack(x_batch)
    fgsm_report = analyzer.metrics.calculate_metrics(x_batch, x_fgsm, np.array([pred]))
    
    # PGD metrics
    x_pgd = analyzer.level1.pgd_attack(x_batch)
    pgd_report = analyzer.metrics.calculate_metrics(x_batch, x_pgd, np.array([pred]))
    
    # Patch metrics
    x_patch, patch_pos = analyzer.level3.apply_adversarial_patch(x)
    patch_report = analyzer.metrics.calculate_metrics(x_batch, x_patch[np.newaxis, ...], np.array([pred]))
    
    # Defense results
    defense_results = analyzer.level4.evaluate_defense(x_batch, x_pgd, pred)
    
    # Generate the HTML report
    report_path = generate_html_report(
        image_name=image_name,
        clean_pred=pred,
        clean_conf=conf,
        noise_slider_results=slider_results,
        fgsm_report=fgsm_report,
        pgd_report=pgd_report,
        patch_report=patch_report,
        defense_results=defense_results,
        output_dir=".",
        dashboard_path="aravm_dashboard.png",
        heatmap_path="level1_heatmap_comparison.png"
    )
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("   📊 aravm_dashboard.png        - Visual attack comparison")
    print("   🔥 level1_heatmap_comparison.png - Gradient heatmaps")
    print("   📄 security_report.html       - Full HTML report")
    print(f"\n🌐 Open the report: file://{os.path.abspath(report_path)}")
    
    return analyzer, reports, report_path

if __name__ == "__main__":
    analyzer, reports, report_path = main()
