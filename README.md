# CS614 Deep Learning for Medical Image Classification

This repository contains comprehensive experiments on transfer learning for histopathology image classification, specifically focusing on cross-domain transfer from breast cancer (BreakHis) to bone cancer (Osteosarcoma) datasets.

## 🏆 Key Results

- **Best Overall Performance**: MobileNetV2 Direct Training - **96.12% accuracy**
- **Best Cross-Domain Transfer**: ResNet50 Full CNN Transfer - **93.16% accuracy**
- **Most Robust Model**: DenseNet121 - **91.45% cross-domain accuracy**

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

## 🔬 Experiments Overview

### Datasets
- **Source**: BreakHis (Breast cancer histopathology, 7,909 images, 2 classes)
- **Target**: Osteosarcoma (Bone cancer histopathology, 867 images, 3 classes)

### Models Tested
- **ResNet50** (23.5M parameters)
- **MobileNetV2** (2.2M parameters) 
- **DenseNet121** (7.0M parameters)

### Transfer Learning Strategies
1. **Direct Training**: Train from scratch on target dataset
2. **Partial Fine-tuning**: Freeze early layers, train last N layers
3. **Full CNN Transfer**: Train entire CNN on source, use as feature extractor
4. **Feature Extraction**: Freeze CNN, train only classifier

### Domain Adaptation Techniques
- **Image-level**: Stain normalization (Reinhard, Macenko, Vahadane)
- **Feature-level**: Batch/Layer/Instance normalization, Moment matching

## 📊 Key Findings

### What Worked ✅
- **Direct training** achieved best results (95%+ accuracy)
- **Advanced optimization** (AdamW, CosineAnnealing) improved performance by 6-7%
- **Feature-level alignment** outperformed image-level stain normalization
- **Full CNN training** beat partial fine-tuning for cross-domain transfer
- **DenseNet121** showed best robustness to domain shift

### What Didn't Work ❌
- **Stain normalization** consistently degraded performance (2-17% loss)
- **Image-level domain adaptation** was less effective than feature-level
- **More trainable layers** didn't always improve cross-domain performance

### Performance Gaps
- **Direct vs Cross-domain**: 4-10% accuracy gap
- **No stain norm vs Reinhard**: 7-17% degradation
- **Basic vs Advanced optimization**: 6-7% improvement

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision
pip install scikit-learn matplotlib pillow
pip install kagglehub  # For BreakHis dataset
```

### Run Best Performing Models

**For Direct Training (Best Overall):**
```bash
python mobilenet_trainer.py  # 96.12% accuracy
python resnet_trainer.py     # 95.73% accuracy
```

**For Cross-Domain Transfer:**
```bash
python full_cnn_transfer_trainer.py     # 93.16% accuracy
python breakhis_transfer_trainer.py     # 91.45% accuracy
```

### Data Setup
1. **Osteosarcoma Dataset**: Place in `osteosarcoma_organized/` folder
2. **BreakHis Dataset**: Automatically downloaded via kagglehub

## 📈 Results Summary

| Approach | Best Model | Test Accuracy | Use Case |
|----------|------------|---------------|----------|
| **Direct Training** | MobileNetV2 | **96.12%** | Sufficient target data |
| **Full CNN Transfer** | ResNet50 | **93.16%** | Cross-domain with large source |
| **Partial Fine-tuning** | DenseNet121 | **91.45%** | Limited target data |
| **Feature Extraction** | DenseNet121 | **91.45%** | Fast training needed |

## 🔧 Advanced Features

### Optimization Techniques
- **AdamW optimizer** with weight decay
- **CosineAnnealingWarmRestarts** scheduler
- **Label smoothing** (0.1)
- **Gradient clipping** (max_norm=1.0)
- **Multi-layer classifiers** with BatchNorm

### Domain Adaptation
- **Stain Normalization**: Reinhard, Macenko, Vahadane methods
- **Feature Alignment**: Batch/Layer/Instance normalization
- **Statistical Matching**: Moment matching for domain alignment

## 📝 Detailed Results

For comprehensive experimental results, analysis, and comparisons, see:
- [`experimental_results.md`](experimental_results.md) - Complete results summary
- [`script_comparison.md`](script_comparison.md) - Detailed script comparison

## 🎯 Recommendations

### For New Projects:
1. **Start with direct training** if you have sufficient target data
2. **Use MobileNetV2** for best efficiency/accuracy trade-off
3. **Apply advanced optimization** (AdamW, CosineAnnealing, Label smoothing)
4. **Avoid stain normalization** for cross-tissue-type transfer
5. **Use feature-level alignment** over image-level domain adaptation

### For Cross-Domain Transfer:
1. **ResNet50 + Full CNN training** for maximum performance
2. **DenseNet121** for robust cross-domain transfer
3. **Feature-level domain alignment** (BatchNorm/LayerNorm)
4. **1-2 trainable layers** optimal for partial fine-tuning

## 📚 Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{cs614_medical_transfer_learning,
  title={Deep Learning Transfer Learning for Medical Image Classification: BreakHis to Osteosarcoma},
  author={CS614 Project},
  year={2024},
  note={Comprehensive experimental study on cross-domain transfer learning for histopathology image classification}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Best Achieved Accuracy: 96.12%** (MobileNetV2 Direct Training)  
**Total Experiments: 50+ individual model training runs**  
**Scripts: 10+ different training configurations**
