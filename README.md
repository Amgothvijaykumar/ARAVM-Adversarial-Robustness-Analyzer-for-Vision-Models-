# 🛡️ ARAVM - Adversarial Robustness Analyzer for Vision Models

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A comprehensive toolkit for evaluating AI vision model security against adversarial attacks**

[Abstract](#-abstract) • [Why It Matters](#-why-it-matters) • [Applications](#-real-world-applications) • [Quick Start](#-quick-start) • [Features](#-features)

</div>

---

## 📄 Abstract

Modern AI vision models powering critical systems—from autonomous vehicles reading traffic signs to facial recognition securing buildings—are vulnerable to **adversarial attacks**. These attacks involve carefully crafted perturbations (invisible noise, patches, or occlusions) that cause AI models to misclassify inputs while appearing unchanged to human observers.

**ARAVM (Adversarial Robustness Analyzer for Vision Models)** is a unified security auditing toolkit that:

1. **Evaluates** the robustness of image classification and face recognition models
2. **Demonstrates** multiple attack vectors (FGSM, PGD, Patch/ROA attacks)
3. **Quantifies** vulnerabilities through comprehensive metrics (misclassification rate, distortion measures)
4. **Visualizes** attack effects through heatmaps and before/after comparisons
5. **Tests defenses** (JPEG compression, spatial smoothing) against attacks
6. **Generates reports** (HTML) summarizing findings and recommendations

This toolkit integrates three powerful libraries:
- **IBM ART** (Adversarial Robustness Toolbox) - Attack implementations
- **Captum** - Interpretability and attribution analysis
- **phattacks** - Rectangular Occlusion Attack (ROA) for patch attacks

---

## ⚠️ Why It Matters

### The Security Problem

| System | Vulnerability | Real-World Impact |
|--------|--------------|-------------------|
| 🚗 **Self-driving cars** | Misread STOP signs | Accidents, loss of life |
| 👤 **Face unlock** | Bypass authentication | Unauthorized access |
| 🏥 **Medical AI** | Wrong diagnosis | Incorrect treatment |
| 📹 **Security cameras** | Evade detection | Criminal activity |
| ✈️ **Airport scanners** | Hide weapons | Security breach |

### Key Statistics
- **97%** of AI models tested are vulnerable to adversarial attacks (MIT Study, 2023)
- **$50 million+** estimated annual losses from AI security breaches
- **< 1%** pixel change needed to fool most image classifiers

### Why ARAVM?
> Before deploying any AI vision system in security-critical applications, you must test its robustness. ARAVM provides a standardized, comprehensive framework to identify vulnerabilities before attackers do.

---

## 🌍 Real-World Applications

### 1. Autonomous Vehicles 🚗
```
Attack: Sticker on STOP sign → AI reads "Speed Limit 80"
Defense: Test with ARAVM before deployment
```

### 2. Facial Recognition Systems 👤
```
Attack: Special glasses/makeup → Become "invisible" to cameras
Defense: Test recognition robustness with live analyzer
```

### 3. Medical Imaging 🏥
```
Attack: Add noise to X-ray → Hide cancer / Create fake tumor
Defense: Validate AI diagnosis with adversarial testing
```

### 4. Document Verification 📄
```
Attack: Modify check images → Change ₹1,000 to ₹1,00,000
Defense: Test OCR systems against perturbations
```

### 5. Biometric Security 🔐
```
Attack: Adversarial face mask → Impersonate another person
Defense: Evaluate face matching under attack conditions
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎚️ **Noise Slider** | Adjust attack intensity (ε) from subtle to aggressive |
| 🔥 **Heatmap Visualization** | See where the model focuses before/after attack |
| 📊 **Robustness Metrics** | MR, L2/L∞ distortion, confidence change |
| 📄 **HTML Reports** | Professional security audit reports |
| 👤 **Live Face Analysis** | Real-time webcam-based attack demo |
| 🛡️ **Defense Testing** | Evaluate JPEG, smoothing defenses |

---

## � Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Amgothvijaykumar/ARAVM-Adversarial-Robustness-Analyzer-for-Vision-Models-.git
cd ARAVM-Adversarial-Robustness-Analyzer-for-Vision-Models-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0,<2.0
opencv-contrib-python>=4.8.0
pillow>=9.0.0
matplotlib>=3.5.0
adversarial-robustness-toolbox>=1.20.0
captum>=0.6.0
scipy>=1.10.0
PyYAML>=6.0
```

---

## 🚀 How to Run

### Step-by-Step Instructions

#### 1️⃣ Clone and Setup
```bash
# Clone the repository
git clone https://github.com/Amgothvijaykumar/ARAVM-Adversarial-Robustness-Analyzer-for-Vision-Models-.git
cd ARAVM-Adversarial-Robustness-Analyzer-for-Vision-Models-

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

#### 2️⃣ Run Image Classification Security Audit

**Option A: Quick Test with Your Own Image**
```bash
# Place your image in the datasets folder, then run:
python run_with_image.py datasets/your_image.jpg
```

**Option B: Full Security Audit (All Attack Levels)**
```bash
python main_analyzer.py
```

This will:
- Load a pre-trained ResNet-50 model
- Run Level 1-4 security attacks (FGSM, PGD, ROA, etc.)
- Generate `security_report.html` with full analysis
- Create visualization images (`aravm_dashboard.png`, heatmaps)

#### 3️⃣ Run Face Recognition Security Audit

```bash
# Step 1: Download face detection models
cd facedetection
python download_models.py
cd ..

# Step 2: Create folders for team members
mkdir -p facedetection/group_members/YourName

# Step 3: Capture face photos (takes 3-5 photos per person)
python capture_faces.py

# Step 4: Train the face recognition model
python facedetection/train_face_recognition.py

# Step 5: Run live face analyzer with adversarial attacks
python live_face_analyzer.py
```

#### 📁 Output Files Generated
| File | Description |
|------|-------------|
| `security_report.html` | 📄 Complete HTML security report |
| `aravm_dashboard.png` | 📊 Visual comparison of all attacks |
| `level1_heatmap_comparison.png` | 🔥 Gradient heatmaps before/after attack |

---

## � Part 1: Image Classification Security Audit

### Option A: Quick Test with Sample Image

```bash
# Activate virtual environment
source venv/bin/activate

# Run with any image
python run_with_image.py datasets/your_image.jpg
```

### Option B: Full Security Audit

```bash
python main_analyzer.py
```

### What Happens:

1. **Level 1 - White-Box Attacks**
   - FGSM (Fast Gradient Sign Method)
   - PGD (Projected Gradient Descent)

2. **Level 2 - Black-Box Attacks**
   - HopSkipJump (query-based)

3. **Level 3 - Patch Attacks**
   - ROA (Rectangular Occlusion Attack)

4. **Level 4 - Defense Evaluation**
   - JPEG Compression
   - Spatial Smoothing

### Output Files:
| File | Description |
|------|-------------|
| `security_report.html` | 📄 Full HTML report with metrics |
| `aravm_dashboard.png` | 📊 Visual attack comparison |
| `level1_heatmap_comparison.png` | 🔥 Gradient heatmaps |

### Example Usage:

```python
from main_analyzer import AdversarialRobustnessAnalyzer, AttackConfig

# Initialize with custom settings
analyzer = AdversarialRobustnessAnalyzer(
    model_name="resnet50",  # or "vgg16", "mobilenet_v2"
    attack_config=AttackConfig(
        epsilon=0.05,       # Noise intensity
        patch_width=50,     # Patch size
        patch_height=50
    )
)

# Load and analyze image
x, y_true = analyzer.load_sample_image("datasets/cat.jpg")

# Run noise slider (test multiple epsilon values)
results = analyzer.noise_slider(x, epsilon_values=[0.01, 0.03, 0.05, 0.1])

# Run full 4-level audit
reports = analyzer.run_full_audit(x, y_true, save_visualizations=True)
```

### Understanding the Output:

```
============================================================
  ROBUSTNESS METRICS REPORT: FGSM (ε=0.03)
============================================================

📊 MODEL CAPABILITY METRICS (Baseline)
  Clean Accuracy:     100.00%    ← Model works on clean images
  Clean Confidence:   85.42%     ← How sure the model is

⚔️  ATTACK EFFECTIVENESS METRICS
  Misclassification Ratio: 100.00%  ← ❌ VULNERABLE! 
  Avg Confidence Change:   0.46     ← Big change = attack worked

📏 PERTURBATION METRICS
  Average L2 Distortion:   11.64    ← Total noise added
  Average L∞ Distortion:   0.03     ← Max change per pixel (3%)
============================================================
```

---

## � Part 2: Face Recognition Security Audit

### Step 1: Download Face Detection Models

```bash
cd facedetection
python download_models.py
cd ..
```

### Step 2: Create Team Member Folders

```bash
mkdir -p facedetection/group_members/YourName
mkdir -p facedetection/group_members/TeamMember2
# Add more as needed
```

### Step 3: Capture Face Photos

```bash
python capture_faces.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `SPACE` | 📸 Take photo |
| `N` | ➡️ Next person |
| `Q` | ❌ Quit |

> **Tip:** Take 3-5 photos per person from different angles!

### Step 4: Train Face Recognition

```bash
python facedetection/train_face_recognition.py
```

**Output:**
```
✓ Models initialized successfully
Processing: Vijay
  ✓ Processed: Vijay_01.jpg - Face at [x, y, w, h]
  ✓ Stored embedding for Vijay (from 3 images)
Processing: Yash
  ✓ Stored embedding for Yash (from 2 images)
...
✓ Training complete! Stored 5 person(s)
```

### Step 5: Run Live Face Analyzer

```bash
python live_face_analyzer.py
```

**Controls:**
| Key | Action | Effect |
|-----|--------|--------|
| `A` | Toggle Adversarial Patch | Colorful patch appears on face |
| `N` | Toggle Gaussian Noise | Invisible noise added to face |
| `P` | Change Patch Position | center → forehead → eyes |
| `+` | Increase Intensity | Stronger attack |
| `-` | Decrease Intensity | Weaker attack |
| `S` | Save Screenshot | Capture current frame |
| `Q` | Quit | Exit program |

### What to Observe:

1. **Without Attack:**
   - Your name appears in GREEN box
   - High confidence score (e.g., 0.85)

2. **With Patch Attack (Press A):**
   - Colorful patch appears on your face
   - Name may change to "Unknown" or wrong person
   - Confidence drops significantly

3. **With Noise Attack (Press N):**
   - Face looks slightly grainy
   - Recognition becomes unreliable
   - Demonstrates FGSM-style attack on faces

---

## � Project Structure

```
ARAVM/
├── main_analyzer.py          # Core analysis engine
├── run_with_image.py         # Run with custom images
├── report_generator.py       # HTML report generator
├── live_face_analyzer.py     # Real-time face attack demo
├── capture_faces.py          # Capture training photos
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── facedetection/
│   ├── models/               # YuNet & SFace ONNX models
│   ├── group_members/        # Team member photos
│   │   ├── Vijay/
│   │   ├── Yash/
│   │   └── ...
│   ├── train_face_recognition.py
│   ├── inference_face_recognition.py
│   └── download_models.py
│
├── datasets/                 # Test images
│   ├── cat.jpg
│   ├── stop.png
│   └── ...
│
├── Captum/                   # Interpretability examples
├── adversarial-robustness-toolbox/  # ART examples
└── phattacks/                # Patch attack implementations
```

---

## 📊 Metrics Explained

| Metric | What It Measures | Good if... |
|--------|------------------|------------|
| **Clean Accuracy** | Model performance on unmodified images | High (close to 100%) |
| **Misclassification Ratio (MR)** | % of images fooled by attack | Low (model is robust) |
| **L2 Distortion** | Total magnitude of perturbation | High (attack is visible) |
| **L∞ Distortion** | Maximum change to any pixel | High (attack is obvious) |
| **Confidence Change** | How much prediction confidence shifts | Low (model is stable) |

---

## �️ Recommended Mitigations

Based on ARAVM findings, implement these defenses:

1. **Adversarial Training** - Retrain model on adversarial examples
2. **Input Preprocessing** - JPEG compression, smoothing
3. **Ensemble Methods** - Use multiple models together
4. **Certified Defenses** - Randomized smoothing for guarantees
5. **Runtime Detection** - Monitor for unusual inputs

---

## 👥 Team

- **Vijay** - Project Lead
- **Yash** - Development
- **Shruthi** - Testing
- **Tharun** - Documentation
- **Vamc** - Research

---

## 📚 References

1. Goodfellow, I. J., et al. "Explaining and Harnessing Adversarial Examples." ICLR 2015.
2. Madry, A., et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018.
3. IBM Adversarial Robustness Toolbox: https://github.com/Trusted-AI/adversarial-robustness-toolbox
4. Captum: https://captum.ai/
5. OpenCV Face Detection: https://github.com/opencv/opencv_zoo

---

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for AI Security Research**

🛡️ *Making AI systems more robust, one vulnerability at a time* 🛡️

</div>
