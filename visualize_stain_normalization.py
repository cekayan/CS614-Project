#!/usr/bin/env python3
"""
Stain Normalization Visualization Script
=======================================

This script creates side-by-side comparisons of images with and without
Reinhard stain normalization for both BreakHis and Osteosarcoma datasets.

It will show:
1. Original BreakHis images vs Reinhard-normalized BreakHis images
2. Original Osteosarcoma images vs Reinhard-normalized Osteosarcoma images
3. How the normalization makes BreakHis look more like Osteosarcoma
"""

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

# Import our components
from stain_normalization import ReinhardNormalizer
from breakhis_transfer_trainer import (
    download_breakhis_dataset, create_breakhis_dataset_from_raw, BreakHisDataset
)
from dataloader import create_dataloaders

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization."""
    # Handle different tensor formats
    if torch.is_tensor(tensor):
        # Denormalize if normalized with ImageNet stats
        if tensor.min() < 0:  # Likely normalized
            # Denormalize using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            
            if tensor.dim() == 3:
                mean = mean.view(3, 1, 1)
                std = std.view(3, 1, 1)
            
            tensor = tensor * std + mean
        
        # Scale to 0-255 if needed
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        
        # Convert from CHW to HWC format
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        return tensor.clamp(0, 255).byte().numpy()
    else:
        # Already numpy array
        if isinstance(tensor, np.ndarray):
            if tensor.max() <= 1.0:
                tensor = tensor * 255.0
            return tensor.astype(np.uint8)
        else:
            # PIL Image
            return np.array(tensor)

def create_comparison_plot(images_dict, title, save_path):
    """Create a comparison plot showing original vs normalized images."""
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Column headers
    col_headers = ['BreakHis Original', 'BreakHis Normalized', 'Osteosarcoma Original', 'Osteosarcoma Normalized']
    for i, header in enumerate(col_headers):
        axes[0, i].set_title(header, fontsize=12, fontweight='bold')
    
    # Row labels
    row_labels = ['Example 1', 'Example 2', 'Example 3']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=12, fontweight='bold')
    
    # Plot images
    for row in range(3):
        # BreakHis Original
        axes[row, 0].imshow(images_dict['breakhis_original'][row])
        axes[row, 0].axis('off')
        
        # BreakHis Normalized
        axes[row, 1].imshow(images_dict['breakhis_normalized'][row])
        axes[row, 1].axis('off')
        
        # Osteosarcoma Original
        axes[row, 2].imshow(images_dict['osteo_original'][row])
        axes[row, 2].axis('off')
        
        # Osteosarcoma Normalized
        axes[row, 3].imshow(images_dict['osteo_normalized'][row])
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved main visualization: {save_path}")
    plt.close()  # Close to free memory
    
    # Create individual plots for better viewing
    for row in range(3):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Sample {row+1} - Stain Normalization Comparison', fontsize=14)
        
        axes[0].imshow(images_dict['breakhis_original'][row])
        axes[0].set_title('BreakHis Original')
        axes[0].axis('off')
        
        axes[1].imshow(images_dict['breakhis_normalized'][row])
        axes[1].set_title('BreakHis Normalized')
        axes[1].axis('off')
        
        axes[2].imshow(images_dict['osteo_original'][row])
        axes[2].set_title('Osteosarcoma Original')
        axes[2].axis('off')
        
        axes[3].imshow(images_dict['osteo_normalized'][row])
        axes[3].set_title('Osteosarcoma Normalized')
        axes[3].axis('off')
        
        individual_save_path = f'./stain_comparison_sample_{row+1}.png'
        plt.tight_layout()
        plt.savefig(individual_save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved individual sample {row+1}: {individual_save_path}")
        plt.close()  # Close to free memory

def get_sample_images(dataset, num_samples=3):
    """Get random sample images from dataset."""
    indices = random.sample(range(len(dataset)), num_samples)
    images = []
    
    for idx in indices:
        image, label = dataset[idx]
        
        # Convert tensor to numpy for visualization
        if torch.is_tensor(image):
            image_np = tensor_to_numpy(image)
        else:
            image_np = np.array(image)
        
        images.append(image_np)
    
    return images

def apply_stain_norm_to_samples(images, normalizer):
    """Apply stain normalization to sample images."""
    normalized_images = []
    
    for img in images:
        try:
            # Ensure image is in proper format
            if isinstance(img, np.ndarray) and img.dtype == np.uint8:
                # Convert to RGB if needed
                if len(img.shape) == 3 and img.shape[2] == 3:
                    normalized = normalizer.normalize(img)
                    normalized_images.append(normalized)
                else:
                    print("Warning: Skipping image with unexpected shape")
                    normalized_images.append(img)
            else:
                print("Warning: Image not in expected uint8 format")
                normalized_images.append(img)
                
        except Exception as e:
            print(f"Warning: Stain normalization failed: {e}")
            normalized_images.append(img)
    
    return normalized_images

def main():
    """Main visualization pipeline."""
    print("🎨 STAIN NORMALIZATION VISUALIZATION")
    print("=" * 50)
    
    # Set random seed for reproducible samples
    random.seed(42)
    torch.manual_seed(42)
    
    # Step 1: Load datasets
    print("\n📥 Loading datasets...")
    
    # Load BreakHis dataset
    print("Loading BreakHis dataset...")
    raw_dataset_path = download_breakhis_dataset()
    if raw_dataset_path is None:
        print("❌ Failed to download BreakHis dataset")
        return
    
    all_images, _, _ = create_breakhis_dataset_from_raw(raw_dataset_path)
    if all_images is None:
        print("❌ Failed to parse BreakHis dataset structure")
        return
    
    # Create BreakHis dataset (without heavy augmentation for visualization)
    simple_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Get sample paths and labels
    all_paths = []
    all_labels = []
    label_map = {'benign': 0, 'malignant': 1}
    
    for class_name, paths in all_images.items():
        sample_paths = random.sample(paths, min(50, len(paths)))  # Get 50 samples per class
        all_paths.extend(sample_paths)
        all_labels.extend([label_map[class_name]] * len(sample_paths))
    
    breakhis_dataset = BreakHisDataset(all_paths, all_labels, simple_transforms)
    
    # Load Osteosarcoma dataset
    print("Loading Osteosarcoma dataset...")
    osteo_dataloaders, osteo_sizes, osteo_classes = create_dataloaders('./osteosarcoma_organized', batch_size=1)
    osteo_dataset = osteo_dataloaders['train'].dataset
    
    print(f"BreakHis samples: {len(breakhis_dataset)}")
    print(f"Osteosarcoma samples: {len(osteo_dataset)}")
    
    # Step 2: Fit Reinhard normalizer
    print("\n🔬 Fitting Reinhard normalizer...")
    
    # Get a representative target image (Osteosarcoma)
    target_image, _ = osteo_dataset[0]
    target_image_np = tensor_to_numpy(target_image)
    
    # Create and fit normalizer
    normalizer = ReinhardNormalizer()
    normalizer.fit(target_image_np)
    print("✅ Reinhard normalizer fitted to Osteosarcoma reference image")
    
    # Step 3: Get sample images
    print("\n🖼️  Collecting sample images...")
    
    # Get samples from both datasets
    breakhis_samples = get_sample_images(breakhis_dataset, num_samples=3)
    osteo_samples = get_sample_images(osteo_dataset, num_samples=3)
    
    print(f"Collected {len(breakhis_samples)} BreakHis samples")
    print(f"Collected {len(osteo_samples)} Osteosarcoma samples")
    
    # Step 4: Apply stain normalization
    print("\n🎨 Applying Reinhard stain normalization...")
    
    breakhis_normalized = apply_stain_norm_to_samples(breakhis_samples, normalizer)
    osteo_normalized = apply_stain_norm_to_samples(osteo_samples, normalizer)
    
    # Step 5: Create visualization
    print("\n📊 Creating visualization...")
    
    images_dict = {
        'breakhis_original': breakhis_samples,
        'breakhis_normalized': breakhis_normalized,
        'osteo_original': osteo_samples,
        'osteo_normalized': osteo_normalized
    }
    
    # Create the comparison plot
    create_comparison_plot(
        images_dict, 
        'Reinhard Stain Normalization: BreakHis → Osteosarcoma Style',
        './stain_normalization_comparison.png'
    )
    
    # Step 6: Print analysis
    print("\n📈 ANALYSIS:")
    print("-" * 50)
    print("👀 What to look for:")
    print("1. BreakHis Original vs Normalized:")
    print("   - Color shifts (pink/purple → blue/purple)")
    print("   - Contrast changes")
    print("   - Overall 'look' becoming more like Osteosarcoma")
    print()
    print("2. Osteosarcoma Original vs Normalized:")
    print("   - Minimal changes (self-normalization)")
    print("   - Slight color adjustments")
    print()
    print("3. Cross-domain comparison:")
    print("   - BreakHis Normalized should look more similar to Osteosarcoma Original")
    print("   - This is the intended effect of stain normalization")
    
    # Step 7: Color statistics comparison
    print("\n📊 COLOR STATISTICS COMPARISON:")
    print("-" * 50)
    
    def compute_color_stats(images, name):
        """Compute mean RGB values for a set of images."""
        all_pixels = []
        for img in images:
            if len(img.shape) == 3:
                pixels = img.reshape(-1, 3)
                all_pixels.append(pixels)
        
        if all_pixels:
            combined = np.vstack(all_pixels)
            mean_rgb = np.mean(combined, axis=0)
            std_rgb = np.std(combined, axis=0)
            print(f"{name:<25} Mean RGB: [{mean_rgb[0]:6.1f}, {mean_rgb[1]:6.1f}, {mean_rgb[2]:6.1f}]")
            print(f"{name:<25} Std RGB:  [{std_rgb[0]:6.1f}, {std_rgb[1]:6.1f}, {std_rgb[2]:6.1f}]")
    
    compute_color_stats(breakhis_samples, "BreakHis Original")
    compute_color_stats(breakhis_normalized, "BreakHis Normalized")
    compute_color_stats(osteo_samples, "Osteosarcoma Original")
    compute_color_stats(osteo_normalized, "Osteosarcoma Normalized")
    
    print("\n✅ Visualization completed!")
    print("📁 Saved: ./stain_normalization_comparison.png")
    
    # Final summary display
    print("\n" + "="*60)
    print("🖼️  VISUAL SUMMARY")
    print("="*60)
    print("The visualization shows:")
    print("• Column 1: BreakHis (breast cancer) original images")
    print("• Column 2: BreakHis images transformed to Osteosarcoma color style")
    print("• Column 3: Osteosarcoma (bone cancer) original images") 
    print("• Column 4: Osteosarcoma images with self-normalization")
    print("\n🎯 Key observation:")
    print("Column 2 (BreakHis Normalized) should look more similar to")
    print("Column 3 (Osteosarcoma Original) in terms of color/staining.")
    print("\nThis explains why your models perform better WITHOUT stain")
    print("normalization - the transformation may remove important features!")
    
    print("\n📁 GENERATED VISUALIZATION FILES:")
    print("- stain_normalization_comparison.png (main 3x4 grid)")
    print("- stain_comparison_sample_1.png (individual sample 1)")
    print("- stain_comparison_sample_2.png (individual sample 2)")
    print("- stain_comparison_sample_3.png (individual sample 3)")
    print("\n💡 You can view these PNG files in your file browser or IDE!")
    
    return images_dict

if __name__ == "__main__":
    images = main()
