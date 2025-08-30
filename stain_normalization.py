"""
Stain Normalization Methods for Histopathology Cross-Domain Transfer Learning

This module implements various stain normalization techniques specifically designed
for histopathology images to address domain shift between different datasets.

Methods included:
1. Reinhard Color Normalization
2. Macenko Stain Normalization  
3. Vahadane Stain Normalization
4. Simple Color Transfer
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import lstsq
import warnings
warnings.filterwarnings('ignore')

class ReinhardNormalizer:
    """
    Reinhard Color Normalization for histopathology images.
    
    Reference: 
    Reinhard et al. "Color Transfer between Images" IEEE Computer Graphics and Applications, 2001
    
    This method transfers color statistics from a target image to source images
    by matching mean and standard deviation in LAB color space.
    """
    
    def __init__(self, target_image=None):
        self.target_stats = None
        if target_image is not None:
            self.fit(target_image)
    
    def rgb_to_lab(self, image):
        """Convert RGB image to LAB color space."""
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to LAB
        lab_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        return lab_image.astype(np.float32)
    
    def lab_to_rgb(self, lab_image):
        """Convert LAB image back to RGB."""
        lab_image = np.clip(lab_image, 0, 255)
        rgb_image = cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return rgb_image.astype(np.float32) / 255.0
    
    def fit(self, target_image):
        """Compute target image statistics in LAB space."""
        if torch.is_tensor(target_image):
            if target_image.dim() == 4:  # Batch dimension
                target_image = target_image[0]
            # Convert CHW to HWC
            if target_image.dim() == 3 and target_image.shape[0] == 3:
                target_image = target_image.permute(1, 2, 0)
            target_image = target_image.cpu().numpy()
        
        # Convert to LAB
        lab_image = self.rgb_to_lab(target_image)
        
        # Compute statistics for each channel
        self.target_stats = {
            'mean': np.mean(lab_image.reshape(-1, 3), axis=0),
            'std': np.std(lab_image.reshape(-1, 3), axis=0)
        }
        
        print(f"Target LAB stats - Mean: {self.target_stats['mean']}, Std: {self.target_stats['std']}")
    
    def normalize(self, source_image):
        """Apply Reinhard normalization to source image."""
        if self.target_stats is None:
            raise ValueError("Must fit normalizer with target image first!")
        
        original_format = None
        if torch.is_tensor(source_image):
            original_format = 'tensor'
            if source_image.dim() == 4:  # Batch dimension
                source_image = source_image[0]
            # Convert CHW to HWC
            if source_image.dim() == 3 and source_image.shape[0] == 3:
                source_image = source_image.permute(1, 2, 0)
            source_image = source_image.cpu().numpy()
        
        # Convert to LAB
        lab_source = self.rgb_to_lab(source_image)
        
        # Compute source statistics
        lab_flat = lab_source.reshape(-1, 3)
        source_mean = np.mean(lab_flat, axis=0)
        source_std = np.std(lab_flat, axis=0)
        
        # Normalize source to target statistics
        normalized_lab = np.zeros_like(lab_source)
        for i in range(3):
            if source_std[i] > 0:
                normalized_lab[:, :, i] = (
                    (lab_source[:, :, i] - source_mean[i]) / source_std[i] 
                    * self.target_stats['std'][i] + self.target_stats['mean'][i]
                )
            else:
                normalized_lab[:, :, i] = lab_source[:, :, i]
        
        # Convert back to RGB
        normalized_rgb = self.lab_to_rgb(normalized_lab)
        
        # Return in original format
        if original_format == 'tensor':
            normalized_rgb = torch.from_numpy(normalized_rgb).permute(2, 0, 1)
            
        return normalized_rgb

class MacenkoNormalizer:
    """
    Macenko Stain Normalization for H&E histopathology images.
    
    Reference:
    Macenko et al. "A method for normalizing histology slides for quantitative analysis" 
    ISBI 2009
    
    This method separates H&E stains using optical density and normalizes based on
    reference stain concentrations.
    """
    
    def __init__(self, target_image=None):
        self.target_stain_matrix = None
        self.target_percentiles = None
        
        # H&E stain vectors (typical values)
        self.he_ref = np.array([[0.5626, 0.2159],
                                [0.7201, 0.8012],
                                [0.4062, 0.5581]])
        
        if target_image is not None:
            self.fit(target_image)
    
    def rgb_to_od(self, image):
        """Convert RGB to Optical Density."""
        # Add small epsilon to avoid log(0)
        image = np.maximum(image, 1e-6)
        return -np.log(image)
    
    def od_to_rgb(self, od):
        """Convert Optical Density to RGB."""
        return np.exp(-od)
    
    def normalize_rows(self, A):
        """Normalize rows of matrix A."""
        return A / (np.linalg.norm(A, axis=1)[:, None] + 1e-12)
    
    def fit(self, target_image):
        """Extract reference stain matrix from target image."""
        if torch.is_tensor(target_image):
            if target_image.dim() == 4:
                target_image = target_image[0]
            if target_image.dim() == 3 and target_image.shape[0] == 3:
                target_image = target_image.permute(1, 2, 0)
            target_image = target_image.cpu().numpy()
        
        # Ensure [0, 1] range
        if target_image.max() > 1.0:
            target_image = target_image / 255.0
        
        # Convert to OD
        od = self.rgb_to_od(target_image).reshape(-1, 3)
        
        # Remove transparent pixels
        od = od[np.sum(od, axis=1) > 0.15]
        
        # Compute eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(np.cov(od.T))
            # Sort by eigenvalue
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]
            
            # Project OD on plane
            proj = od @ eigvecs[:, :2]
            
            # Find extreme angles
            angles = np.arctan2(proj[:, 1], proj[:, 0])
            min_angle = np.percentile(angles, 1)
            max_angle = np.percentile(angles, 99)
            
            # Convert back to OD vectors
            v1 = eigvecs[:, :2] @ [np.cos(min_angle), np.sin(min_angle)]
            v2 = eigvecs[:, :2] @ [np.cos(max_angle), np.sin(max_angle)]
            
            # Ensure proper orientation
            if v1[0] > v2[0]:
                self.target_stain_matrix = np.array([v1, v2]).T
            else:
                self.target_stain_matrix = np.array([v2, v1]).T
                
            # Normalize
            self.target_stain_matrix = self.normalize_rows(self.target_stain_matrix.T).T
            
            # Compute percentiles for each stain
            concentrations = lstsq(self.target_stain_matrix, od.T)[0]
            self.target_percentiles = np.percentile(concentrations, 99, axis=1)
            
        except Exception as e:
            print(f"Warning: Macenko fitting failed, using default H&E matrix: {e}")
            self.target_stain_matrix = self.he_ref[:, :2]
            self.target_percentiles = np.array([1.5, 1.0])
    
    def normalize(self, source_image):
        """Apply Macenko normalization to source image."""
        if self.target_stain_matrix is None:
            raise ValueError("Must fit normalizer with target image first!")
        
        original_format = None
        if torch.is_tensor(source_image):
            original_format = 'tensor'
            if source_image.dim() == 4:
                source_image = source_image[0]
            if source_image.dim() == 3 and source_image.shape[0] == 3:
                source_image = source_image.permute(1, 2, 0)
            source_image = source_image.cpu().numpy()
        
        # Ensure [0, 1] range
        if source_image.max() > 1.0:
            source_image = source_image / 255.0
        
        h, w = source_image.shape[:2]
        
        # Convert to OD
        od = self.rgb_to_od(source_image).reshape(-1, 3)
        
        # Separate stains
        try:
            concentrations = lstsq(self.target_stain_matrix, od.T)[0]
            
            # Normalize concentrations
            source_percentiles = np.percentile(concentrations, 99, axis=1)
            normalized_concentrations = concentrations * (
                self.target_percentiles[:, None] / (source_percentiles[:, None] + 1e-12)
            )
            
            # Reconstruct
            normalized_od = self.target_stain_matrix @ normalized_concentrations
            normalized_rgb = self.od_to_rgb(normalized_od.T).reshape(h, w, 3)
            
            # Clip to valid range
            normalized_rgb = np.clip(normalized_rgb, 0, 1)
            
        except Exception as e:
            print(f"Warning: Macenko normalization failed, returning original: {e}")
            normalized_rgb = source_image
        
        # Return in original format
        if original_format == 'tensor':
            normalized_rgb = torch.from_numpy(normalized_rgb).permute(2, 0, 1)
            
        return normalized_rgb

class VahadaneNormalizer:
    """
    Vahadane Stain Normalization using Non-negative Matrix Factorization.
    
    Reference:
    Vahadane et al. "Structure-Preserving Color Normalization and Sparse Stain Separation 
    for Histological Images" IEEE TMI 2016
    
    More robust method using sparsity constraints.
    """
    
    def __init__(self, target_image=None, regularizer=0.1):
        self.target_stain_matrix = None
        self.target_maxC = None
        self.regularizer = regularizer
        
        if target_image is not None:
            self.fit(target_image)
    
    def rgb_to_od(self, image):
        """Convert RGB to Optical Density."""
        image = np.maximum(image, 1e-6)
        return -np.log(image)
    
    def od_to_rgb(self, od):
        """Convert Optical Density to RGB."""
        return np.exp(-od)
    
    def simple_nmf(self, V, n_components=2, max_iter=100):
        """Simple Non-negative Matrix Factorization implementation."""
        m, n = V.shape
        W = np.random.rand(m, n_components)
        H = np.random.rand(n_components, n)
        
        for _ in range(max_iter):
            # Update H
            H = H * ((W.T @ V) / (W.T @ W @ H + 1e-12))
            
            # Update W  
            W = W * ((V @ H.T) / (W @ H @ H.T + 1e-12))
            
            # Add sparsity regularization to H
            H = np.maximum(H - self.regularizer, 0)
        
        return W, H
    
    def fit(self, target_image):
        """Extract reference stain matrix using NMF."""
        if torch.is_tensor(target_image):
            if target_image.dim() == 4:
                target_image = target_image[0]
            if target_image.dim() == 3 and target_image.shape[0] == 3:
                target_image = target_image.permute(1, 2, 0)
            target_image = target_image.cpu().numpy()
        
        # Ensure [0, 1] range
        if target_image.max() > 1.0:
            target_image = target_image / 255.0
        
        # Convert to OD
        od = self.rgb_to_od(target_image).reshape(-1, 3)
        
        # Remove transparent pixels
        od = od[np.sum(od, axis=1) > 0.15]
        
        try:
            # Apply NMF to separate stains
            W, H = self.simple_nmf(od.T, n_components=2)
            
            # Normalize stain vectors
            self.target_stain_matrix = W / (np.linalg.norm(W, axis=0)[None, :] + 1e-12)
            
            # Compute max concentrations
            concentrations = H
            self.target_maxC = np.percentile(concentrations, 99, axis=1)
            
        except Exception as e:
            print(f"Warning: Vahadane fitting failed, using default: {e}")
            # Fallback to typical H&E values
            self.target_stain_matrix = np.array([[0.5626, 0.2159],
                                                [0.7201, 0.8012], 
                                                [0.4062, 0.5581]])
            self.target_maxC = np.array([1.5, 1.0])
    
    def normalize(self, source_image):
        """Apply Vahadane normalization to source image."""
        if self.target_stain_matrix is None:
            raise ValueError("Must fit normalizer with target image first!")
        
        original_format = None
        if torch.is_tensor(source_image):
            original_format = 'tensor'
            if source_image.dim() == 4:
                source_image = source_image[0]
            if source_image.dim() == 3 and source_image.shape[0] == 3:
                source_image = source_image.permute(1, 2, 0)
            source_image = source_image.cpu().numpy()
        
        # Ensure [0, 1] range
        if source_image.max() > 1.0:
            source_image = source_image / 255.0
        
        h, w = source_image.shape[:2]
        
        # Convert to OD
        od = self.rgb_to_od(source_image).reshape(-1, 3)
        
        try:
            # Separate stains using NMF
            W, H = self.simple_nmf(od.T, n_components=2)
            
            # Normalize source stain matrix
            W_norm = W / (np.linalg.norm(W, axis=0)[None, :] + 1e-12)
            
            # Compute source max concentrations
            source_maxC = np.percentile(H, 99, axis=1)
            
            # Normalize concentrations
            H_normalized = H * (self.target_maxC[:, None] / (source_maxC[:, None] + 1e-12))
            
            # Reconstruct with target stain matrix
            normalized_od = self.target_stain_matrix @ H_normalized
            normalized_rgb = self.od_to_rgb(normalized_od.T).reshape(h, w, 3)
            
            # Clip to valid range
            normalized_rgb = np.clip(normalized_rgb, 0, 1)
            
        except Exception as e:
            print(f"Warning: Vahadane normalization failed, returning original: {e}")
            normalized_rgb = source_image
        
        # Return in original format
        if original_format == 'tensor':
            normalized_rgb = torch.from_numpy(normalized_rgb).permute(2, 0, 1)
            
        return normalized_rgb

class StainNormalizationTransform:
    """
    PyTorch-compatible transform for stain normalization.
    Can be integrated into existing data loading pipelines.
    """
    
    def __init__(self, method='reinhard', target_image=None, **kwargs):
        """
        Args:
            method: 'reinhard', 'macenko', or 'vahadane'
            target_image: Reference image for normalization
            **kwargs: Additional parameters for specific methods
        """
        self.method = method.lower()
        
        if method == 'reinhard':
            self.normalizer = ReinhardNormalizer(target_image)
        elif method == 'macenko':
            self.normalizer = MacenkoNormalizer(target_image)
        elif method == 'vahadane':
            self.normalizer = VahadaneNormalizer(target_image, **kwargs)
        else:
            raise ValueError(f"Unknown stain normalization method: {method}")
    
    def __call__(self, image):
        """Apply stain normalization to input image."""
        return self.normalizer.normalize(image)
    
    def fit(self, target_image):
        """Fit normalizer to target image."""
        self.normalizer.fit(target_image)

def compare_stain_normalization_methods(source_image, target_image, save_path=None):
    """
    Compare different stain normalization methods side by side.
    
    Args:
        source_image: Image to be normalized
        target_image: Reference image for normalization
        save_path: Optional path to save comparison plot
    """
    methods = {
        'Reinhard': ReinhardNormalizer(target_image),
        'Macenko': MacenkoNormalizer(target_image),
        'Vahadane': VahadaneNormalizer(target_image)
    }
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(source_image)
    axes[0, 0].set_title('Source Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target_image)
    axes[0, 1].set_title('Target Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')  # Empty
    
    # Normalized results
    for i, (name, normalizer) in enumerate(methods.items()):
        try:
            normalized = normalizer.normalize(source_image)
            if torch.is_tensor(normalized):
                if normalized.dim() == 3:
                    normalized = normalized.permute(1, 2, 0)
                normalized = normalized.cpu().numpy()
            
            axes[1, i].imshow(np.clip(normalized, 0, 1))
            axes[1, i].set_title(f'{name} Normalized')
            axes[1, i].axis('off')
        except Exception as e:
            axes[1, i].text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f'{name} (Failed)')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
    
    plt.show()
    
    return fig

# Example usage and testing functions
if __name__ == "__main__":
    print("🎨 Stain Normalization Library for Histopathology")
    print("=" * 50)
    print("Available methods:")
    print("1. Reinhard Color Normalization")
    print("2. Macenko Stain Normalization") 
    print("3. Vahadane Stain Normalization")
    print("\nUse these methods to normalize staining variations between")
    print("BreakHis and Osteosarcoma datasets for better cross-domain transfer!")
