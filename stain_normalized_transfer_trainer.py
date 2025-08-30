"""
Stain-Normalized Transfer Learning Script

This script integrates stain normalization with transfer learning for 
cross-domain histopathology classification (BreakHis -> Osteosarcoma).

Combines:
1. Stain normalization (Reinhard/Macenko/Vahadane) at image level
2. Feature-level domain alignment (moment matching)
3. Transfer learning with CNN fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from collections import defaultdict
import kagglehub
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# Import our stain normalization methods
from stain_normalization import (
    ReinhardNormalizer, MacenkoNormalizer, VahadaneNormalizer, 
    StainNormalizationTransform
)

# Import existing components
from breakhis_transfer_trainer import (
    download_breakhis_dataset, create_breakhis_dataset_from_raw, 
    BreakHisDataset, create_breakhis_dataloaders_from_raw
)
from dataloader import create_dataloaders
from full_cnn_transfer_trainer import (
    compute_domain_statistics, apply_feature_normalization
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class StainNormalizedDataset(Dataset):
    """
    Dataset wrapper that applies stain normalization on-the-fly.
    """
    
    def __init__(self, base_dataset, stain_normalizer=None, apply_stain_norm=True):
        """
        Args:
            base_dataset: Original dataset (BreakHis or Osteosarcoma)
            stain_normalizer: Fitted stain normalizer
            apply_stain_norm: Whether to apply stain normalization
        """
        self.base_dataset = base_dataset
        self.stain_normalizer = stain_normalizer
        self.apply_stain_norm = apply_stain_norm
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if self.apply_stain_norm and self.stain_normalizer is not None:
            try:
                # Convert tensor to format expected by normalizer
                if torch.is_tensor(image):
                    # Ensure image is in [0, 1] range
                    if image.max() > 1.0:
                        image = image / 255.0
                    
                    # Apply stain normalization
                    normalized_image = self.stain_normalizer.normalize(image)
                    
                    # Ensure output is tensor
                    if not torch.is_tensor(normalized_image):
                        normalized_image = torch.from_numpy(normalized_image)
                    
                    image = normalized_image
                    
            except Exception as e:
                print(f"Warning: Stain normalization failed for sample {idx}: {e}")
                # Use original image if normalization fails
                pass
        
        return image, label

class StainNormalizedResNet(nn.Module):
    """ResNet50 with stain normalization integration."""
    
    def __init__(self, num_classes=3, freeze_features=False, enable_feature_alignment=True):
        super(StainNormalizedResNet, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.enable_feature_alignment = enable_feature_alignment
        
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        # Feature alignment buffers
        if enable_feature_alignment:
            self.register_buffer('source_mean', torch.zeros(num_features))
            self.register_buffer('source_std', torch.ones(num_features))
            self.register_buffer('target_mean', torch.zeros(num_features))
            self.register_buffer('target_std', torch.ones(num_features))
    
    def set_domain_statistics(self, source_stats, target_stats):
        """Set statistics for feature-level domain alignment."""
        if self.enable_feature_alignment:
            source_mean, source_std = source_stats
            target_mean, target_std = target_stats
            
            self.source_mean.copy_(source_mean)
            self.source_std.copy_(source_std)
            self.target_mean.copy_(target_mean)
            self.target_std.copy_(target_std)
    
    def get_features(self, x):
        """Extract features without classification."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x, apply_feature_alignment=False):
        """Forward pass with optional feature alignment."""
        # Extract features
        features = self.get_features(x)
        
        # Apply feature-level domain alignment if enabled
        if apply_feature_alignment and self.enable_feature_alignment:
            features = apply_feature_normalization(
                features, (self.source_mean, self.source_std),
                (self.target_mean, self.target_std)
            )
        
        # Classification
        return self.resnet.fc(features)

def create_stain_normalized_dataloaders(base_dataloaders, stain_normalizer, 
                                       normalize_splits=['train', 'val', 'test']):
    """
    Create stain-normalized versions of dataloaders.
    
    Args:
        base_dataloaders: Original dataloaders dict
        stain_normalizer: Fitted stain normalizer
        normalize_splits: Which splits to apply normalization to
    
    Returns:
        Dictionary of stain-normalized dataloaders
    """
    normalized_dataloaders = {}
    
    for split, dataloader in base_dataloaders.items():
        apply_norm = split in normalize_splits
        
        # Wrap dataset with stain normalization
        normalized_dataset = StainNormalizedDataset(
            dataloader.dataset, stain_normalizer, apply_norm
        )
        
        # Create new dataloader
        normalized_dataloaders[split] = DataLoader(
            normalized_dataset,
            batch_size=dataloader.batch_size,
            shuffle=(split == 'train'),
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )
    
    return normalized_dataloaders

def fit_stain_normalizer_from_datasets(source_dataset, target_dataset, method='reinhard', 
                                      n_samples=50):
    """
    Fit stain normalizer using sample images from both datasets.
    
    Args:
        source_dataset: Source domain dataset (BreakHis)
        target_dataset: Target domain dataset (Osteosarcoma)
        method: Stain normalization method
        n_samples: Number of samples to use for fitting
    
    Returns:
        Fitted stain normalizer
    """
    print(f"Fitting {method} stain normalizer...")
    
    # Sample random images from target dataset
    target_indices = np.random.choice(len(target_dataset), 
                                     min(n_samples, len(target_dataset)), 
                                     replace=False)
    
    # Get a representative target image (use first sample)
    target_image, _ = target_dataset[target_indices[0]]
    
    # Create and fit normalizer
    if method.lower() == 'reinhard':
        normalizer = ReinhardNormalizer()
    elif method.lower() == 'macenko':
        normalizer = MacenkoNormalizer()
    elif method.lower() == 'vahadane':
        normalizer = VahadaneNormalizer()
    else:
        raise ValueError(f"Unknown stain normalization method: {method}")
    
    # Fit to target image
    normalizer.fit(target_image)
    
    print(f"Stain normalizer fitted using target dataset sample")
    return normalizer

def train_with_stain_normalization(model, train_loader, val_loader, device, 
                                  num_epochs=10, apply_feature_alignment=False):
    """
    Train model with stain-normalized data.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, apply_feature_alignment=apply_feature_alignment)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, apply_feature_alignment=apply_feature_alignment)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')
    
    return model, train_losses, val_accuracies, best_val_acc

def evaluate_comprehensive(model, dataloader, device, class_names, 
                          apply_feature_alignment=False, phase_name='Test'):
    """
    Comprehensive evaluation with all metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, apply_feature_alignment=apply_feature_alignment)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate metrics
    accuracy = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'predictions': all_predictions,
        'labels': all_labels
    }

def main():
    """Main training and evaluation pipeline with stain normalization."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    stain_methods = ['reinhard', 'macenko', 'vahadane', 'none']  # 'none' = no stain norm
    feature_alignment_options = [True, False]  # With/without feature alignment
    
    results = defaultdict(dict)
    
    print("🎨 Loading datasets...")
    
    # Load BreakHis dataset
    breakhis_path = download_breakhis_dataset()
    breakhis_dataloaders = create_breakhis_dataloaders_from_raw(breakhis_path, batch_size=32)
    
    # Load Osteosarcoma dataset
    osteo_dataloaders = create_dataloaders('/home/cek99/CS614', batch_size=32)
    
    print("📊 Starting comprehensive stain normalization experiments...")
    print("=" * 70)
    
    for stain_method in stain_methods:
        for use_feature_alignment in feature_alignment_options:
            
            config_name = f"{stain_method}_stain_{'with' if use_feature_alignment else 'without'}_feature_align"
            print(f"\n🧪 Configuration: {config_name}")
            print("-" * 50)
            
            try:
                # Create stain normalizer
                if stain_method != 'none':
                    stain_normalizer = fit_stain_normalizer_from_datasets(
                        breakhis_dataloaders['train'].dataset,
                        osteo_dataloaders['train'].dataset,
                        method=stain_method
                    )
                    
                    # Apply stain normalization to BreakHis data
                    normalized_breakhis_loaders = create_stain_normalized_dataloaders(
                        breakhis_dataloaders, stain_normalizer, 
                        normalize_splits=['train', 'val', 'test']
                    )
                else:
                    # No stain normalization
                    normalized_breakhis_loaders = breakhis_dataloaders
                    stain_normalizer = None
                
                # Create model
                model = StainNormalizedResNet(
                    num_classes=4,  # BreakHis has 4 classes
                    freeze_features=False,
                    enable_feature_alignment=use_feature_alignment
                ).to(device)
                
                # Phase 1: Train on normalized BreakHis data
                print("Phase 1: Training on BreakHis (stain normalized)...")
                model, train_losses, val_accs, best_val_acc = train_with_stain_normalization(
                    model, normalized_breakhis_loaders['train'], 
                    normalized_breakhis_loaders['val'], device, 
                    num_epochs=5, apply_feature_alignment=False
                )
                
                print(f"BreakHis training completed - Best Val Acc: {best_val_acc:.2f}%")
                
                # Phase 2: Prepare for transfer to Osteosarcoma
                print("Phase 2: Preparing transfer to Osteosarcoma...")
                
                # Freeze feature extractor and replace classifier
                for param in model.resnet.parameters():
                    param.requires_grad = False
                
                # Replace classifier for Osteosarcoma (3 classes)
                num_features = model.resnet.fc[1].in_features
                model.resnet.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 3)  # Osteosarcoma has 3 classes
                ).to(device)
                
                # Set up feature alignment if enabled
                if use_feature_alignment:
                    print("Setting up feature-level domain alignment...")
                    
                    # Create temporary model for feature extraction
                    feature_extractor = StainNormalizedResNet(num_classes=4, freeze_features=True)
                    feature_extractor.load_state_dict(model.state_dict(), strict=False)
                    feature_extractor.eval()
                    
                    # Compute domain statistics
                    source_stats = compute_domain_statistics(
                        normalized_breakhis_loaders['train'], feature_extractor, device
                    )
                    target_stats = compute_domain_statistics(
                        osteo_dataloaders['train'], feature_extractor, device
                    )
                    model.set_domain_statistics(source_stats, target_stats)
                    
                    print("Feature alignment statistics computed")
                
                # Apply stain normalization to Osteosarcoma if using same method
                if stain_method != 'none':
                    normalized_osteo_loaders = create_stain_normalized_dataloaders(
                        osteo_dataloaders, stain_normalizer,
                        normalize_splits=['train', 'val', 'test']
                    )
                else:
                    normalized_osteo_loaders = osteo_dataloaders
                
                # Phase 3: Fine-tune on Osteosarcoma
                print("Phase 3: Fine-tuning on Osteosarcoma...")
                model, _, _, osteo_best_val_acc = train_with_stain_normalization(
                    model, normalized_osteo_loaders['train'],
                    normalized_osteo_loaders['val'], device,
                    num_epochs=10, apply_feature_alignment=use_feature_alignment
                )
                
                # Phase 4: Final evaluation
                print("Phase 4: Final evaluation...")
                osteo_classes = ['0', '1', '2']  # Assuming 3 classes
                
                train_results = evaluate_comprehensive(
                    model, normalized_osteo_loaders['train'], device, osteo_classes,
                    apply_feature_alignment=use_feature_alignment, phase_name='Train'
                )
                
                val_results = evaluate_comprehensive(
                    model, normalized_osteo_loaders['val'], device, osteo_classes,
                    apply_feature_alignment=use_feature_alignment, phase_name='Validation'
                )
                
                test_results = evaluate_comprehensive(
                    model, normalized_osteo_loaders['test'], device, osteo_classes,
                    apply_feature_alignment=use_feature_alignment, phase_name='Test'
                )
                
                # Store results
                results[config_name] = {
                    'breakhis_val_acc': best_val_acc,
                    'osteo_train': train_results,
                    'osteo_val': val_results,
                    'osteo_test': test_results
                }
                
                # Print results for this configuration
                print(f"\n✅ Results for {config_name}:")
                print(f"BreakHis Val Accuracy: {best_val_acc:.2f}%")
                print("Osteosarcoma Results:")
                print(f"  Train - Acc: {train_results['accuracy']:.2f}%, F1: {train_results['f1']:.2f}%")
                print(f"  Val   - Acc: {val_results['accuracy']:.2f}%, F1: {val_results['f1']:.2f}%")
                print(f"  Test  - Acc: {test_results['accuracy']:.2f}%, F1: {test_results['f1']:.2f}%")
                
            except Exception as e:
                print(f"❌ Error in configuration {config_name}: {e}")
                results[config_name] = {'error': str(e)}
    
    # Final comparison table
    print("\n" + "="*100)
    print("🏆 FINAL COMPARISON TABLE - STAIN NORMALIZATION + TRANSFER LEARNING")
    print("="*100)
    
    print(f"{'Configuration':<40} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 100)
    
    for config_name, result in results.items():
        if 'error' not in result:
            train_acc = result['osteo_train']['accuracy']
            val_acc = result['osteo_val']['accuracy']
            test_acc = result['osteo_test']['accuracy']
            test_f1 = result['osteo_test']['f1']
            
            print(f"{config_name:<40} {train_acc:<12.2f} {val_acc:<12.2f} {test_acc:<12.2f} {test_f1:<12.2f}")
        else:
            print(f"{config_name:<40} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
    
    print("="*100)
    print("🎯 Key Insights:")
    print("1. Compare stain normalization methods (Reinhard vs Macenko vs Vahadane)")
    print("2. Evaluate stain normalization + feature alignment combination")
    print("3. Identify best approach for BreakHis → Osteosarcoma transfer")
    
    return results

if __name__ == "__main__":
    results = main()
