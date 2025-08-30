#!/usr/bin/env python3
"""
Simple Stain Normalization Transfer Learning
==========================================

This script tests ONLY stain normalization with basic transfer learning:
- No advanced optimizers (just Adam)
- No label smoothing
- No feature alignment 
- No advanced data augmentation
- No gradient clipping
- Just: CNN + basic training + stain normalization (or not)

The goal: Isolate the effect of stain normalization alone.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import time
import copy
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

# Import our components
from stain_normalization import ReinhardNormalizer
from breakhis_transfer_trainer import (
    download_breakhis_dataset, create_breakhis_dataset_from_raw, BreakHisDataset
)
from dataloader import create_dataloaders
from sklearn.metrics import precision_recall_fscore_support

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class StainNormalizedDataset(Dataset):
    """Simple dataset wrapper for stain normalization."""
    
    def __init__(self, base_dataset, stain_normalizer=None, apply_stain_norm=True):
        self.base_dataset = base_dataset
        self.stain_normalizer = stain_normalizer
        self.apply_stain_norm = apply_stain_norm
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if self.apply_stain_norm and self.stain_normalizer is not None:
            try:
                # Convert tensor to numpy for stain normalization
                if torch.is_tensor(image):
                    # Denormalize first if normalized
                    if image.min() < 0:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        image = image * std + mean
                    
                    if image.max() <= 1.0:
                        image = image * 255.0
                    
                    # Convert CHW to HWC
                    image_np = image.permute(1, 2, 0).clamp(0, 255).byte().numpy()
                else:
                    image_np = np.array(image)
                
                # Apply stain normalization
                normalized_image = self.stain_normalizer.normalize(image_np)
                
                # Convert back to tensor
                if not torch.is_tensor(normalized_image):
                    normalized_image = torch.from_numpy(normalized_image)
                
                # Convert back to CHW and normalize
                if normalized_image.dim() == 3:
                    normalized_image = normalized_image.permute(2, 0, 1)
                
                normalized_image = normalized_image.float() / 255.0
                
                # Apply ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = (normalized_image - mean) / std
                
            except Exception as e:
                print(f"Warning: Stain normalization failed for sample {idx}: {e}")
                # Keep original image if normalization fails
        
        return image, label

class SimpleResNet(nn.Module):
    """Simple ResNet50 with basic classifier - no advanced features."""
    
    def __init__(self, num_classes=3, trainable_layers=1):
        super(SimpleResNet, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        
        # Freeze all layers first
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze specified number of last layers
        layer_names = ['layer4', 'layer3', 'layer2', 'layer1']
        for i in range(min(trainable_layers, len(layer_names))):
            layer = getattr(self.resnet, layer_names[i])
            for param in layer.parameters():
                param.requires_grad = True
        
        # Simple classifier - just dropout + linear
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        print(f"  → Trainable CNN layers: {layer_names[:trainable_layers]}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  → Parameters: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}% trainable)")
    
    def forward(self, x):
        return self.resnet(x)

def create_simple_transforms():
    """Simple data transforms - basic resize and normalization only."""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms

def simple_train(model, dataloaders, dataset_sizes, device, num_epochs=10):
    """Simple training loop - no advanced techniques."""
    print(f"🔥 Simple training for {num_epochs} epochs...")
    
    # Basic loss and optimizer
    criterion = nn.CrossEntropyLoss()  # No label smoothing
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Basic Adam
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Simple scheduler
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'  Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()  # No gradient clipping
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'  {phase.capitalize()}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.4f}')
            
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f'  Best validation accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def simple_evaluate(model, dataloader, dataset_size, device, split_name):
    """Simple evaluation - return accuracy, precision, recall, F1."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / dataset_size
    acc = running_corrects.double() / dataset_size
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    print(f'{split_name}: Loss {loss:.4f}, Acc {acc:.4f}, Precision {precision:.4f}, Recall {recall:.4f}, F1 {f1:.4f}')
    
    return {
        'loss': loss,
        'accuracy': acc.item(),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def run_experiment(stain_method, trainable_layers=1):
    """Run a single experiment with or without stain normalization."""
    
    print(f"\n{'='*80}")
    print(f"🧪 EXPERIMENT: {stain_method.upper()} + {trainable_layers} TRAINABLE LAYER(S)")
    print(f"{'='*80}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    print("\n📥 Loading datasets...")
    
    # BreakHis
    raw_dataset_path = download_breakhis_dataset()
    if raw_dataset_path is None:
        return None
        
    all_images, _, _ = create_breakhis_dataset_from_raw(raw_dataset_path)
    if all_images is None:
        return None
    
    # Create BreakHis dataset with simple transforms
    train_transforms, val_test_transforms = create_simple_transforms()
    
    all_paths = []
    all_labels = []
    label_map = {'benign': 0, 'malignant': 1}
    
    for class_name, paths in all_images.items():
        all_paths.extend(paths)
        all_labels.extend([label_map[class_name]] * len(paths))
    
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    
    total_samples = len(all_paths)
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)
    
    train_dataset = BreakHisDataset(all_paths[:train_end], all_labels[:train_end], train_transforms)
    val_dataset = BreakHisDataset(all_paths[train_end:val_end], all_labels[train_end:val_end], val_test_transforms)
    
    breakhis_dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4),
        'validation': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    }
    breakhis_sizes = {'train': len(train_dataset), 'validation': len(val_dataset)}
    
    # Osteosarcoma
    osteo_dataloaders, osteo_sizes, osteo_classes = create_dataloaders('./osteosarcoma_organized', batch_size=16)
    
    print(f"BreakHis: {len(all_images['benign'])} benign + {len(all_images['malignant'])} malignant")
    print(f"Osteosarcoma: {osteo_classes} (sizes: {osteo_sizes})")
    
    # Set up stain normalization if needed
    stain_normalizer = None
    if stain_method == "reinhard":
        print("\n🎨 Setting up Reinhard stain normalization...")
        
        # Get reference image from Osteosarcoma
        target_image, _ = osteo_dataloaders['train'].dataset[0]
        
        # Convert to numpy
        if torch.is_tensor(target_image):
            if target_image.min() < 0:  # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                target_image = target_image * std + mean
            
            if target_image.max() <= 1.0:
                target_image = target_image * 255.0
            
            target_image_np = target_image.permute(1, 2, 0).clamp(0, 255).byte().numpy()
        else:
            target_image_np = np.array(target_image)
        
        # Fit normalizer
        stain_normalizer = ReinhardNormalizer()
        stain_normalizer.fit(target_image_np)
        print("✅ Reinhard normalizer fitted")
        
        # Apply to BreakHis data
        breakhis_dataloaders = {
            split: DataLoader(
                StainNormalizedDataset(dataloader.dataset, stain_normalizer, True),
                batch_size=dataloader.batch_size,
                shuffle=(split == 'train'),
                num_workers=dataloader.num_workers
            )
            for split, dataloader in breakhis_dataloaders.items()
        }
        
        # Apply to Osteosarcoma data  
        osteo_dataloaders = {
            split: DataLoader(
                StainNormalizedDataset(dataloader.dataset, stain_normalizer, True),
                batch_size=dataloader.batch_size,
                shuffle=(split == 'train'),
                num_workers=dataloader.num_workers
            )
            for split, dataloader in osteo_dataloaders.items()
        }
    
    # Phase 1: Train on BreakHis
    print(f"\n📚 Phase 1: Training on BreakHis")
    model = SimpleResNet(num_classes=2, trainable_layers=trainable_layers).to(device)
    model = simple_train(model, breakhis_dataloaders, breakhis_sizes, device, num_epochs=10)
    
    # Phase 2: Transfer to Osteosarcoma
    print(f"\n🔄 Phase 2: Transfer to Osteosarcoma")
    
    # Freeze BreakHis-trained features
    for name, param in model.named_parameters():
        if 'layer' in name:
            param.requires_grad = False
    
    # Replace classifier for 3 classes
    num_features = model.resnet.fc[1].in_features  # Get from Linear layer
    model.resnet.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, len(osteo_classes))
    ).to(device)
    
    # Train classifier on Osteosarcoma
    model = simple_train(model, osteo_dataloaders, osteo_sizes, device, num_epochs=20)
    
    # Phase 3: Evaluate
    print(f"\n📊 Final Evaluation:")
    results = {}
    for split in ['train', 'validation', 'test']:
        results[split] = simple_evaluate(
            model, osteo_dataloaders[split], osteo_sizes[split], device, split.capitalize()
        )
    
    return results

def main():
    """Main comparison: None vs Reinhard stain normalization."""
    print("🧪 SIMPLE STAIN NORMALIZATION COMPARISON")
    print("=" * 80)
    print("Testing ONLY stain normalization effect:")
    print("- No advanced optimizers")
    print("- No feature alignment") 
    print("- No advanced augmentation")
    print("- Just basic transfer learning + stain norm (or not)")
    
    experiments = [
        ("none", 1),
        ("reinhard", 1)
    ]
    
    all_results = {}
    
    for stain_method, trainable_layers in experiments:
        try:
            results = run_experiment(stain_method, trainable_layers)
            if results:
                all_results[f"{stain_method}_{trainable_layers}layer"] = results
        except Exception as e:
            print(f"❌ Error in {stain_method}: {e}")
            all_results[f"{stain_method}_{trainable_layers}layer"] = {'error': str(e)}
    
    # Final comparison
    print(f"\n{'='*80}")
    print("🏆 SIMPLE TRANSFER LEARNING RESULTS")
    print(f"{'='*80}")
    print(f"{'Configuration':<20} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Test F1':<10}")
    print("-" * 70)
    
    for config_name, results in all_results.items():
        if 'error' not in results:
            train_acc = results['train']['accuracy']
            val_acc = results['validation']['accuracy']
            test_acc = results['test']['accuracy']
            test_f1 = results['test']['f1']
            
            print(f"{config_name:<20} {train_acc:<10.4f} {val_acc:<10.4f} {test_acc:<10.4f} {test_f1:<10.4f}")
        else:
            print(f"{config_name:<20} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
    
    print("-" * 70)
    
    # Calculate improvement
    if 'none_1layer' in all_results and 'reinhard_1layer' in all_results:
        if 'error' not in all_results['none_1layer'] and 'error' not in all_results['reinhard_1layer']:
            none_acc = all_results['none_1layer']['test']['accuracy']
            reinhard_acc = all_results['reinhard_1layer']['test']['accuracy']
            
            improvement = reinhard_acc - none_acc
            print(f"\n📈 Stain Normalization Effect:")
            print(f"Baseline (None):    {none_acc:.4f} ({none_acc*100:.1f}%)")
            print(f"Reinhard:           {reinhard_acc:.4f} ({reinhard_acc*100:.1f}%)")
            print(f"Difference:         {improvement:+.4f} ({improvement*100:+.1f}%)")
            
            if improvement > 0:
                print("✅ Stain normalization HELPS!")
            else:
                print("❌ Stain normalization HURTS!")
    
    return all_results

if __name__ == "__main__":
    results = main()
