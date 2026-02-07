"""
ARAVM HTML Report Generator
Generates a comprehensive HTML report summarizing vulnerabilities and mitigations
"""

import os
import base64
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 for embedding in HTML"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""


def generate_html_report(
    image_name: str,
    clean_pred: int,
    clean_conf: float,
    noise_slider_results: Dict,
    fgsm_report,
    pgd_report,
    patch_report,
    defense_results: Dict,
    output_dir: str = ".",
    dashboard_path: str = "aravm_dashboard.png",
    heatmap_path: str = "level1_heatmap_comparison.png"
) -> str:
    """
    Generate comprehensive HTML security report
    
    Returns: Path to generated HTML report
    """
    
    # Convert images to base64
    dashboard_b64 = image_to_base64(os.path.join(output_dir, dashboard_path))
    heatmap_b64 = image_to_base64(os.path.join(output_dir, heatmap_path))
    
    # Calculate overall vulnerability score (0-100, higher = more vulnerable)
    fgsm_vuln = fgsm_report.misclassification_ratio * 100
    pgd_vuln = pgd_report.misclassification_ratio * 100
    patch_vuln = patch_report.misclassification_ratio * 100
    
    overall_vulnerability = (fgsm_vuln * 0.3 + pgd_vuln * 0.5 + patch_vuln * 0.2)
    
    # Determine risk level
    if overall_vulnerability >= 70:
        risk_level = "CRITICAL"
        risk_color = "#dc3545"
        risk_icon = "🔴"
    elif overall_vulnerability >= 40:
        risk_level = "HIGH"
        risk_color = "#fd7e14"
        risk_icon = "🟠"
    elif overall_vulnerability >= 20:
        risk_level = "MEDIUM"
        risk_color = "#ffc107"
        risk_icon = "🟡"
    else:
        risk_level = "LOW"
        risk_color = "#28a745"
        risk_icon = "🟢"
    
    # Generate noise slider table rows
    noise_rows = ""
    for eps, result in noise_slider_results.items():
        status = "✗ FOOLED" if result['success'] else "✓ Robust"
        status_class = "status-fail" if result['success'] else "status-pass"
        noise_rows += f"""
        <tr>
            <td>ε = {eps:.2f}</td>
            <td>{result['prediction']}</td>
            <td>{result['confidence']:.2%}</td>
            <td>{result['l2_distortion']:.3f}</td>
            <td class="{status_class}">{status}</td>
        </tr>
        """
    
    # Defense effectiveness rows
    defense_rows = ""
    for defense_name, result in defense_results.items():
        status = "✓ RECOVERED" if result.get('correct', False) else "✗ Still Fooled"
        status_class = "status-pass" if result.get('correct', False) else "status-fail"
        defense_rows += f"""
        <tr>
            <td>{defense_name.replace('_', ' ').title()}</td>
            <td>{result.get('confidence', 0):.2%}</td>
            <td class="{status_class}">{status}</td>
        </tr>
        """
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARAVM Security Report - {image_name}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --danger: #dc3545;
            --warning: #ffc107;
            --success: #28a745;
            --dark: #1e293b;
            --light: #f8fafc;
            --gray: #64748b;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            text-align: center;
            padding: 3rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 2rem;
        }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--gray);
            font-size: 1.1rem;
        }}
        
        .timestamp {{
            color: var(--gray);
            font-size: 0.9rem;
            margin-top: 1rem;
        }}
        
        .card {{
            background: rgba(30, 41, 59, 0.8);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .card-title {{
            font-size: 1.3rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .risk-badge {{
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: bold;
            font-size: 1.2rem;
            background: {risk_color};
            color: white;
            margin: 1rem 0;
        }}
        
        .score-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 2rem;
            padding: 2rem;
        }}
        
        .score-circle {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: conic-gradient({risk_color} {overall_vulnerability}%, #334155 0);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        
        .score-inner {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: #1e293b;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }}
        
        .score-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: {risk_color};
        }}
        
        .score-label {{
            font-size: 0.8rem;
            color: var(--gray);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        th {{
            background: rgba(37, 99, 235, 0.2);
            font-weight: 600;
        }}
        
        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        .status-pass {{
            color: var(--success);
            font-weight: bold;
        }}
        
        .status-fail {{
            color: var(--danger);
            font-weight: bold;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .metric-box {{
            background: rgba(37, 99, 235, 0.1);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #60a5fa;
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: var(--gray);
        }}
        
        .image-container {{
            text-align: center;
            margin: 1rem 0;
        }}
        
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .mitigation-list {{
            list-style: none;
            padding: 0;
        }}
        
        .mitigation-list li {{
            padding: 0.75rem;
            margin: 0.5rem 0;
            background: rgba(40, 167, 69, 0.1);
            border-left: 4px solid var(--success);
            border-radius: 0 8px 8px 0;
        }}
        
        .mitigation-list li strong {{
            color: var(--success);
        }}
        
        .attack-summary {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .attack-card {{
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }}
        
        .attack-card.fgsm {{ background: linear-gradient(135deg, rgba(253, 126, 20, 0.2), rgba(253, 126, 20, 0.05)); }}
        .attack-card.pgd {{ background: linear-gradient(135deg, rgba(220, 53, 69, 0.2), rgba(220, 53, 69, 0.05)); }}
        .attack-card.patch {{ background: linear-gradient(135deg, rgba(102, 16, 242, 0.2), rgba(102, 16, 242, 0.05)); }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--gray);
            border-top: 1px solid rgba(255,255,255,0.1);
            margin-top: 2rem;
        }}
        
        @media (max-width: 768px) {{
            .attack-summary {{
                grid-template-columns: 1fr;
            }}
            .score-container {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛡️ ARAVM Security Report</h1>
            <p class="subtitle">Adversarial Robustness Analyzer for Vision Models</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <!-- Executive Summary -->
        <div class="card">
            <h2 class="card-title">📊 Executive Summary</h2>
            <div class="score-container">
                <div>
                    <p><strong>Input Image:</strong> {image_name}</p>
                    <p><strong>Model:</strong> ResNet-50 (ImageNet)</p>
                    <p><strong>Original Prediction:</strong> Class {clean_pred}</p>
                    <p><strong>Original Confidence:</strong> {clean_conf:.2%}</p>
                </div>
                <div class="score-circle">
                    <div class="score-inner">
                        <span class="score-value">{overall_vulnerability:.0f}</span>
                        <span class="score-label">Vulnerability</span>
                    </div>
                </div>
                <div style="text-align: center;">
                    <p>Risk Level:</p>
                    <span class="risk-badge">{risk_icon} {risk_level}</span>
                </div>
            </div>
        </div>
        
        <!-- Attack Summary Cards -->
        <div class="card">
            <h2 class="card-title">⚔️ Attack Effectiveness Summary</h2>
            <div class="attack-summary">
                <div class="attack-card fgsm">
                    <h3>FGSM Attack</h3>
                    <p class="metric-value">{fgsm_report.misclassification_ratio:.0%}</p>
                    <p class="metric-label">Misclassification Rate</p>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem;">L∞ = {fgsm_report.avg_linf_distortion:.4f}</p>
                </div>
                <div class="attack-card pgd">
                    <h3>PGD Attack</h3>
                    <p class="metric-value">{pgd_report.misclassification_ratio:.0%}</p>
                    <p class="metric-label">Misclassification Rate</p>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem;">L∞ = {pgd_report.avg_linf_distortion:.4f}</p>
                </div>
                <div class="attack-card patch">
                    <h3>Patch Attack (ROA)</h3>
                    <p class="metric-value">{patch_report.misclassification_ratio:.0%}</p>
                    <p class="metric-label">Misclassification Rate</p>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem;">L2 = {patch_report.avg_l2_distortion:.2f}</p>
                </div>
            </div>
        </div>
        
        <!-- Visual Dashboard -->
        <div class="card">
            <h2 class="card-title">🖼️ Visual Analysis Dashboard</h2>
            <div class="image-container">
                <img src="data:image/png;base64,{dashboard_b64}" alt="ARAVM Dashboard">
                <p style="color: var(--gray); margin-top: 0.5rem;">Original → FGSM Attack → Patch Attack → Perturbation Magnified</p>
            </div>
        </div>
        
        <!-- Heatmap Comparison -->
        <div class="card">
            <h2 class="card-title">🔥 Gradient Heatmap Analysis</h2>
            <div class="image-container">
                <img src="data:image/png;base64,{heatmap_b64}" alt="Heatmap Comparison">
                <p style="color: var(--gray); margin-top: 0.5rem;">Shows how the model's attention shifts under adversarial attack</p>
            </div>
        </div>
        
        <!-- Noise Slider Analysis -->
        <div class="card">
            <h2 class="card-title">📈 Noise Intensity Analysis (FGSM)</h2>
            <p style="color: var(--gray); margin-bottom: 1rem;">Testing model robustness across different perturbation budgets (ε)</p>
            <table>
                <thead>
                    <tr>
                        <th>Perturbation (ε)</th>
                        <th>Predicted Class</th>
                        <th>Confidence</th>
                        <th>L2 Distortion</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {noise_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Detailed Metrics -->
        <div class="card">
            <h2 class="card-title">📏 Detailed Robustness Metrics</h2>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-value">{fgsm_report.clean_accuracy:.0%}</div>
                    <div class="metric-label">Clean Accuracy</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{fgsm_report.avg_confidence_change:.4f}</div>
                    <div class="metric-label">Avg Confidence Change (FGSM)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{pgd_report.avg_confidence_change:.4f}</div>
                    <div class="metric-label">Avg Confidence Change (PGD)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{pgd_report.avg_l2_distortion:.2f}</div>
                    <div class="metric-label">Avg L2 Distortion (PGD)</div>
                </div>
            </div>
        </div>
        
        <!-- Defense Evaluation -->
        <div class="card">
            <h2 class="card-title">🛡️ Defense Effectiveness</h2>
            <p style="color: var(--gray); margin-bottom: 1rem;">Testing defensive preprocessing against PGD attack</p>
            <table>
                <thead>
                    <tr>
                        <th>Defense Method</th>
                        <th>Confidence</th>
                        <th>Recovery Status</th>
                    </tr>
                </thead>
                <tbody>
                    {defense_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Recommendations -->
        <div class="card">
            <h2 class="card-title">💡 Security Recommendations & Best Practices</h2>
            <ul class="mitigation-list">
                <li>
                    <strong>Adversarial Training:</strong> Retrain the model on adversarial examples generated during this analysis. This is the most effective defense, improving robustness by 40-60% on average.
                </li>
                <li>
                    <strong>Input Preprocessing:</strong> Implement JPEG compression (quality 50-70) or spatial smoothing as a first line of defense. These are computationally cheap and can reduce attack success rate by 15-30%.
                </li>
                <li>
                    <strong>Gradient Masking:</strong> Use defensive distillation or gradient regularization to make gradient-based attacks less effective.
                </li>
                <li>
                    <strong>Ensemble Methods:</strong> Deploy multiple models with different architectures. An adversarial example crafted for one model is less likely to transfer to others.
                </li>
                <li>
                    <strong>Input Validation:</strong> Implement runtime anomaly detection to flag suspicious inputs before classification. Monitor for unusual activation patterns.
                </li>
                <li>
                    <strong>Rate Limiting:</strong> For API deployments, limit query rates to prevent attackers from probing model decision boundaries through iterative queries.
                </li>
                <li>
                    <strong>Certified Defenses:</strong> Consider randomized smoothing for provable robustness guarantees within a certified L2 radius.
                </li>
            </ul>
        </div>
        
        <!-- Technical Details -->
        <div class="card">
            <h2 class="card-title">🔧 Technical Configuration</h2>
            <table>
                <tr><td><strong>Framework</strong></td><td>ARAVM (Adversarial Robustness Analyzer for Vision Models)</td></tr>
                <tr><td><strong>Target Model</strong></td><td>ResNet-50 (ImageNet pre-trained, 1000 classes)</td></tr>
                <tr><td><strong>Attack Library</strong></td><td>IBM Adversarial Robustness Toolbox (ART) v1.20+</td></tr>
                <tr><td><strong>Interpretability</strong></td><td>Captum (Integrated Gradients)</td></tr>
                <tr><td><strong>Patch Attacks</strong></td><td>ROA (Rectangular Occlusion Attack) from phattacks</td></tr>
                <tr><td><strong>FGSM Epsilon</strong></td><td>0.03 (default), tested range: 0.01 - 0.30</td></tr>
                <tr><td><strong>PGD Iterations</strong></td><td>40 steps, step size α = 0.01</td></tr>
                <tr><td><strong>Patch Size</strong></td><td>50 × 50 pixels</td></tr>
            </table>
        </div>
        
        <footer>
            <p>Generated by <strong>ARAVM</strong> - Adversarial Robustness Analyzer for Vision Models</p>
            <p>Integrating ART, Captum, and phattacks for comprehensive security evaluation</p>
            <p style="margin-top: 1rem;">© 2026 ARAVM Framework</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Save the report
    report_path = os.path.join(output_dir, "security_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[✓] HTML Security Report saved: {report_path}")
    return report_path


if __name__ == "__main__":
    # Test with dummy data
    print("Report generator module loaded. Import and call generate_html_report()")
