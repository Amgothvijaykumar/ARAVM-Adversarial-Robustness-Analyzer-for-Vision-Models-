import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class SecurityRiskAnalyzer:
    """
    Analyze and visualize security risks from robustness metrics
    """
    
    @staticmethod
    def plot_robustness_comparison(results: Dict):
        """
        Compare robustness metrics across attribution methods
        """
        methods = list(results.keys())
        infidelity_scores = [results[m]['infidelity'].mean() for m in methods]
        sensitivity_scores = [results[m]['sensitivity'].mean() for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Infidelity comparison
        ax1.barh(methods, infidelity_scores, color='coral', alpha=0.7)
        ax1.set_xlabel('Infidelity Score (lower is better)')
        ax1.set_title('Explanation Faithfulness')
        ax1.axvline(x=0.5, color='red', linestyle='--', label='High Risk')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sensitivity comparison
        ax2.barh(methods, sensitivity_scores, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Sensitivity Score (lower is better)')
        ax2.set_title('Explanation Stability')
        ax2.axvline(x=0.1, color='red', linestyle='--', label='High Risk')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_adversarial_robustness(adv_results: Dict):
        """
        Plot how robustness changes with perturbation strength
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epsilons = adv_results['epsilon']
        
        # Sensitivity vs epsilon
        axes[0].plot(epsilons, adv_results['sensitivity'], 
                    'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Perturbation Radius (ε)')
        axes[0].set_ylabel('Sensitivity Score')
        axes[0].set_title('Explanation Sensitivity vs Perturbation')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # Prediction change vs epsilon
        axes[1].plot(epsilons, adv_results['prediction_change'],
                    'o-', linewidth=2, markersize=8, color='red')
        axes[1].set_xlabel('Perturbation Radius (ε)')
        axes[1].set_ylabel('Prediction Change Rate')
        axes[1].set_title('Prediction Stability')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        # Confidence change vs epsilon
        axes[2].plot(epsilons, adv_results['confidence_change'],
                    'o-', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Perturbation Radius (ε)')
        axes[2].set_ylabel('Confidence Change')
        axes[2].set_title('Confidence Stability')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xscale('log')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_risk_heatmap(risk_assessment: Dict):
        """
        Create a heatmap of security risks
        """
        # Prepare data
        methods = list(risk_assessment.keys())
        metrics = ['Infidelity', 'Sensitivity', 'Overall Risk']
        
        data = []
        for method in methods:
            assess = risk_assessment[method]
            # Normalize scores to 0-1 scale
            infid_norm = min(assess['infidelity'], 1.0)
            sens_norm = min(assess['sensitivity'], 1.0)
            overall = (infid_norm + sens_norm) / 2
            data.append([infid_norm, sens_norm, overall])
        
        df = pd.DataFrame(data, index=methods, columns=metrics)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   vmin=0, vmax=1, cbar_kws={'label': 'Risk Score'},
                   ax=ax)
        ax.set_title('Security Risk Heatmap\n(Higher = More Risky)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_security_report(risk_assessment: Dict) -> str:
        """
        Generate a textual security report
        """
        report = []
        report.append("="*70)
        report.append("SECURITY RISK ASSESSMENT REPORT")
        report.append("="*70)
        report.append("")
        
        # Sort by risk level
        risk_order = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
        sorted_methods = sorted(
            risk_assessment.items(),
            key=lambda x: risk_order.get(x[1]['risk_level'], 0),
            reverse=True
        )
        
        for method, assessment in sorted_methods:
            report.append(f"\n{method}")
            report.append("-" * len(method))
            report.append(f"Risk Level: {assessment['risk_level']}")
            report.append(f"Infidelity Score: {assessment['infidelity']:.6f}")
            report.append(f"Sensitivity Score: {assessment['sensitivity']:.6f}")
            report.append("\nIdentified Risks:")
            for risk in assessment['risks']:
                report.append(f"  • {risk}")
        
        report.append("\n" + "="*70)
        report.append("RECOMMENDATIONS")
        report.append("="*70)
        report.append("")
        
        # Add recommendations based on findings
        critical_count = sum(1 for _, a in risk_assessment.items() 
                           if a['risk_level'] == 'CRITICAL')
        high_count = sum(1 for _, a in risk_assessment.items() 
                        if a['risk_level'] == 'HIGH')
        
        if critical_count > 0:
            report.append("⚠️  CRITICAL: Multiple methods show high vulnerability")
            report.append("   - DO NOT deploy in adversarial environments")
            report.append("   - Implement additional security measures")
            report.append("   - Consider ensemble or robust attribution methods")
        elif high_count > 0:
            report.append("⚠️  WARNING: Some methods show concerning vulnerabilities")
            report.append("   - Exercise caution in deployment")
            report.append("   - Monitor for adversarial inputs")
            report.append("   - Consider using more robust methods")
        else:
            report.append("✓ No critical security risks detected")
            report.append("   - Methods appear reasonably robust")
            report.append("   - Continue monitoring in production")
        
        return "\n".join(report)


# Usage example
analyzer = SecurityRiskAnalyzer()

# Generate visualizations
fig1 = analyzer.plot_robustness_comparison(results)
fig1.savefig('robustness_comparison.png', dpi=300, bbox_inches='tight')

fig2 = analyzer.plot_adversarial_robustness(adv_results)
fig2.savefig('adversarial_robustness.png', dpi=300, bbox_inches='tight')

fig3 = analyzer.create_risk_heatmap(risk_assessment)
fig3.savefig('risk_heatmap.png', dpi=300, bbox_inches='tight')

# Generate report
report = analyzer.generate_security_report(risk_assessment)
print(report)

# Save report
with open('security_assessment_report.txt', 'w') as f:
    f.write(report)