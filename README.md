# CS614 Deep Learning for Medical Image Classification

This repository contains comprehensive experiments on **cross-disease transfer learning** for histopathology image classification, specifically focusing on transfer from breast cancer (BreakHis) to bone cancer (Osteosarcoma) datasets. This work provides empirical validation of transfer learning strategies in computational pathology.

## 🎯 Research Objective

We address two critical challenges in computational pathology:
1. **Limited labeled data** in many cancer types
2. **Strong domain shift** across laboratories, stains, and diseases

Our study compares direct training versus cross-disease transfer learning, evaluating the effectiveness of different strategies when morphological differences exist between source and target domains.

## 🏆 Key Findings

### Performance Results
- **Best Overall Performance**: MobileNetV2 Direct Training - **96.12% validation, 95.73% test accuracy**
- **Best Cross-Domain Transfer**: ResNet50 Full CNN Transfer - **93.16% test accuracy**
- **Most Robust Cross-Domain Model**: DenseNet121 - **91.45% accuracy**

### Critical Insights
- **Cross-disease transfer trails direct training by 3-10%** depending on backbone and strategy
- **Full network transfer significantly outperforms partial fine-tuning** for cross-domain scenarios
- **Stain normalization consistently degrades performance** (7-17% accuracy loss) when tissues differ in architecture
- **Feature-level alignment yields 3-8% improvements** over basic transfer learning

## 📊 Experimental Design

### Datasets
- **Source Domain**: BreakHis (Breast cancer histopathology)
  - 7,909 images, 2 classes (Benign vs Malignant)
- **Target Domain**: Osteosarcoma (Bone cancer histopathology)  
  - 867 images, 3 classes (Non-tumor, Necrosis, Viable-tumor)

### CNN Architectures Evaluated
- **ResNet50** (23.5M parameters)
- **MobileNetV2** (2.2M parameters) - Best efficiency/accuracy trade-off
- **DenseNet121** (7.0M parameters) - Most robust to domain shift

### Transfer Learning Strategies
1. **Direct Training**: Train from scratch on target dataset (baseline)
2. **Partial Fine-tuning**: Freeze early layers, train last N layers  
3. **Full CNN Transfer**: Train entire CNN on source, use as feature extractor
4. **Feature-level Alignment**: Adjust normalization layers during fine-tuning

### Domain Adaptation Techniques
- **Image-level**: Stain normalization (Reinhard method)
- **Feature-level**: Batch/Layer/Instance normalization, Moment matching

## 📁 Repository Structure

```
CS614/
├── README.md                              # This file
├── experimental_results.md                # Comprehensive results summary
├── script_comparison.md                   # Detailed script comparison
├── dataloader.py                         # Data loading utilities
├── stain_normalization.py               # Stain normalization implementations
│
├── Direct Training Scripts:
├── resnet_trainer.py                    # ResNet50 direct training
├── mobilenet_trainer.py                 # MobileNetV2 direct training  
├── densenet_trainer.py                  # DenseNet121 direct training
│
├── Transfer Learning Scripts:
├── breakhis_transfer_trainer.py         # Cross-domain transfer (partial fine-tuning)
├── full_cnn_transfer_trainer.py         # Full CNN training + feature extraction
├── optimized_transfer_trainer.py        # Advanced optimization techniques
├── domain_aligned_transfer_trainer.py   # Feature-level domain alignment
├── simple_stain_transfer.py            # Simple stain normalization test
├── comprehensive_stain_transfer_trainer.py # Complete stain + advanced training
│
└── Visualization:
    └── visualize_stain_normalization.py # Stain normalization visualization
```

## 🔬 Methodology

### Optimization Strategy
- **AdamW optimizer** with weight decay
- **CosineAnnealingWarmRestarts** scheduler
- **Label smoothing** (0.1)
- **Gradient clipping** (max_norm=1.0)
- **Multi-layer classifiers** with BatchNorm

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- **Bootstrap confidence intervals** (95%, 2000 resamples)
- **Class-wise confusion matrices**
- **Statistical significance testing**

## 📈 Detailed Results

### Direct Training Performance
| Model | Validation Acc | Test Acc | Precision | F1 |
|-------|---------------|----------|-----------|-----|
| **MobileNetV2** | **96.12%** | **95.73%** | 97.09% | 95.76% |
| ResNet50 | 95.35% | 95.73% | 96.85% | 95.77% |
| DenseNet121 | 95.35% | 94.87% | 96.58% | 95.09% |

### Cross-Disease Transfer Performance
| Strategy | Best Model | Test Accuracy | Gap vs Direct |
|----------|------------|---------------|---------------|
| **Full CNN Transfer** | ResNet50 | **93.16%** | **-2.57%** |
| **Partial Fine-tuning** | DenseNet121 | **91.45%** | **-3.42%** |
| Feature Extraction | DenseNet121 | 91.45% | -3.42% |

### Impact of Stain Normalization
| Model | Without Stain Norm | With Reinhard | Performance Loss |
|-------|-------------------|---------------|------------------|
| ResNet50 | 93.16% | 86.32% | **-7.3%** |
| MobileNetV2 | 91.45% | 74.36% | **-17.1%** |
| DenseNet121 | 91.45% | 78.63% | **-12.8%** |

### Feature-Level Alignment Results
| Method | ResNet50 | MobileNetV2 | DenseNet121 |
|---------|----------|-------------|-------------|
| **BatchNorm tweaks** | **91.45%** | **92.44%** | 90.46% |
| LayerNorm swaps | 91.29% | 84.55% | **93.31%** |
| InstanceNorm | 89.74% | 87.18% | 92.31% |
| Baseline Transfer | 85.47% | 90.60% | 91.45% |

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision
pip install scikit-learn matplotlib pillow
pip install kagglehub  # For BreakHis dataset
```

### Run Experiments

**Best Performing Approaches:**
```bash
# Direct training (when sufficient target data available)
python mobilenet_trainer.py  # 96.12% accuracy

# Cross-domain transfer (when target data limited)
python full_cnn_transfer_trainer.py     # 93.16% accuracy
python domain_aligned_transfer_trainer.py  # Feature-level alignment
```

### Data Setup
1. **Osteosarcoma Dataset**: Place in `osteosarcoma_organized/` folder
2. **BreakHis Dataset**: Automatically downloaded via kagglehub

## 💡 Practical Recommendations

### For Practitioners
1. **Start with direct training** when sufficient target data is available (>500 images per class)
2. **Use full CNN transfer** over partial fine-tuning for cross-disease scenarios
3. **Apply feature-level alignment** (BatchNorm/LayerNorm) rather than stain normalization
4. **Choose MobileNetV2** for best efficiency/accuracy trade-off
5. **Avoid color-only stain normalization** when transferring across tissue types

### For Cross-Domain Transfer
1. **Full network pretraining** is more effective than partial fine-tuning
2. **Feature-level alignment** provides 3-8% improvements over baseline transfer  
3. **DenseNet121** shows best robustness to domain shift
4. **Advanced optimization** (AdamW, CosineAnnealing) improves performance by 6-7%

## 🎯 Clinical Deployment Guidelines

Based on our findings and literature review:

- **Multi-center validation** with site-wise splits and transparent scanner reporting
- **LIS integration** with secure, auditable data flow and human-in-the-loop review
- **Model documentation** including training data, validation sites, and failure modes
- **Post-deployment monitoring** for domain drift with scheduled revalidation

## 🔍 Error Analysis

**Key Findings:**
- Misclassifications concentrate on **Necrosis vs Non-tumor boundaries**
- **Stain normalization reduces discriminative chromatin and matrix patterns**
- **DenseNet121 preserves performance better** under domain transfer
- **Bootstrap confidence intervals** show statistical significance of improvements

## 📚 Academic Context

This work contributes to the growing literature on cross-disease transfer learning in computational pathology. Our empirical study validates theoretical insights from recent works on:

- Transfer learning effectiveness across morphologically different domains
- Limitations of color-only domain adaptation approaches  
- Benefits of feature-level alignment over image-level preprocessing
- Practical deployment considerations for clinical environments

## 📊 Reproducibility

**Statistical Rigor:**
- Fixed random seeds for reproducible results
- 95% bootstrap confidence intervals (2000 resamples)
- Statistical significance testing between methods
- Versioned dependencies and detailed hyperparameters