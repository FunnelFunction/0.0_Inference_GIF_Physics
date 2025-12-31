"""
shell_extractor.py - Extract Shell Structures (Σ) from Scalar Fields

Shells are the geometric boundaries where the field changes rapidly.
They encode the STRUCTURE of the field - its topology, its organization.

Mathematical Definition:
    Σ = {(i,j) : |∇Φ|(i,j) > τ}
    
Where τ is an adaptive threshold based on field statistics.

In ARC puzzles, shells often correspond to:
    - Boundaries between colored regions
    - Edges of shapes
    - Grid lines
    - Pattern transitions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage

from .scalar_field import ScalarField


@dataclass
class ShellFeatures:
    """Geometric features extracted from shell structure."""
    count: int                    # Number of shell pixels
    coverage: float               # Fraction of field that is shell
    n_components: int             # Number of connected shell regions
    mean_curvature: float         # Average Laplacian at shells
    std_curvature: float          # Curvature variance
    orientation_entropy: float    # How uniform are gradient directions
    thickness: float              # Average shell width
    total_length: float           # Total shell perimeter


class ShellExtractor:
    """
    Extract and analyze shell structures from scalar fields.
    """
    
    def __init__(self, threshold_sigma: float = 1.0):
        """
        Args:
            threshold_sigma: Shells are pixels with |∇Φ| > mean + σ·std
        """
        self.threshold_sigma = threshold_sigma
    
    def extract(self, field: ScalarField) -> np.ndarray:
        """
        Extract shell mask from field.
        
        Returns:
            Boolean mask where True = shell pixel
        """
        grad_mag = field.gradient_magnitude()
        
        # Adaptive threshold
        mean = np.mean(grad_mag.data)
        std = np.std(grad_mag.data)
        threshold = mean + self.threshold_sigma * std
        
        # Handle edge case of uniform field
        if std < 1e-8:
            return np.zeros_like(field.data, dtype=bool)
        
        return grad_mag.data > threshold
    
    def extract_with_threshold(self, field: ScalarField, 
                                threshold: float) -> np.ndarray:
        """Extract shells with explicit threshold."""
        grad_mag = field.gradient_magnitude()
        return grad_mag.data > threshold
    
    def features(self, field: ScalarField, 
                 shell_mask: Optional[np.ndarray] = None) -> ShellFeatures:
        """
        Extract geometric features from shell structure.
        """
        if shell_mask is None:
            shell_mask = self.extract(field)
        
        # Basic counts
        count = int(np.sum(shell_mask))
        total_pixels = shell_mask.size
        coverage = count / total_pixels if total_pixels > 0 else 0.0
        
        # Connected components
        if count > 0:
            labeled, n_components = ndimage.label(shell_mask)
        else:
            n_components = 0
        
        # Curvature at shells
        lap = field.laplacian()
        if count > 0:
            shell_curvatures = lap.data[shell_mask]
            mean_curvature = float(np.mean(shell_curvatures))
            std_curvature = float(np.std(shell_curvatures))
        else:
            mean_curvature = 0.0
            std_curvature = 0.0
        
        # Orientation analysis
        grad_x, grad_y = field.gradient()
        if count > 0:
            angles = np.arctan2(grad_y.data[shell_mask], 
                               grad_x.data[shell_mask])
            # Compute entropy of angle histogram
            hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
            hist = hist / (np.sum(hist) + 1e-8)
            orientation_entropy = -np.sum(hist * np.log(hist + 1e-8))
        else:
            orientation_entropy = 0.0
        
        # Thickness estimation (using distance transform)
        if count > 0:
            dist = ndimage.distance_transform_edt(~shell_mask)
            dist_at_shells = dist[shell_mask]
            thickness = float(np.mean(dist_at_shells)) if len(dist_at_shells) > 0 else 0.0
        else:
            thickness = 0.0
        
        # Total length (perimeter approximation)
        total_length = float(count)  # Simplified: count ≈ length for thin shells
        
        return ShellFeatures(
            count=count,
            coverage=coverage,
            n_components=n_components,
            mean_curvature=mean_curvature,
            std_curvature=std_curvature,
            orientation_entropy=orientation_entropy,
            thickness=thickness,
            total_length=total_length
        )
    
    def curvature_field(self, field: ScalarField) -> ScalarField:
        """
        Compute curvature tensor field ρq.
        
        ρq = ||∇(∇²Φ)||
        
        This measures how the curvature itself varies.
        """
        lap = field.laplacian()
        grad_lap = lap.gradient_magnitude()
        
        result = ScalarField(field.nx, field.ny, field.dx, field.dy)
        result.data = grad_lap.data
        result.name = "ρq"
        
        return result


def shell_distance(Σ1: np.ndarray, Σ2: np.ndarray, 
                   metric: str = 'iou') -> float:
    """
    Compute distance between two shell configurations.
    
    Metrics:
        'iou': 1 - Intersection over Union (0 = identical, 1 = disjoint)
        'dice': 1 - Dice coefficient
        'hausdorff': Maximum closest-point distance
        'l1': Normalized L1 distance
    """
    Σ1 = Σ1.astype(bool)
    Σ2 = Σ2.astype(bool)
    
    if metric == 'iou':
        intersection = np.sum(Σ1 & Σ2)
        union = np.sum(Σ1 | Σ2)
        if union == 0:
            return 0.0  # Both empty = identical
        return 1.0 - intersection / union
    
    elif metric == 'dice':
        intersection = np.sum(Σ1 & Σ2)
        total = np.sum(Σ1) + np.sum(Σ2)
        if total == 0:
            return 0.0
        return 1.0 - 2 * intersection / total
    
    elif metric == 'hausdorff':
        if not np.any(Σ1) or not np.any(Σ2):
            return float(max(Σ1.shape))  # Maximum possible distance
        
        # Distance transform
        dist1 = ndimage.distance_transform_edt(~Σ1)
        dist2 = ndimage.distance_transform_edt(~Σ2)
        
        # Hausdorff = max of directed distances
        h1 = np.max(dist1[Σ2]) if np.any(Σ2) else 0
        h2 = np.max(dist2[Σ1]) if np.any(Σ1) else 0
        
        return max(h1, h2)
    
    elif metric == 'l1':
        return float(np.mean(np.abs(Σ1.astype(float) - Σ2.astype(float))))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def shell_features_distance(f1: ShellFeatures, f2: ShellFeatures) -> float:
    """
    Compute distance between shell feature vectors.
    """
    # Normalize features and compute weighted distance
    features1 = np.array([
        f1.coverage,
        f1.n_components / 100,  # Normalize
        f1.mean_curvature,
        f1.std_curvature,
        f1.orientation_entropy / np.log(16)  # Normalize by max entropy
    ])
    
    features2 = np.array([
        f2.coverage,
        f2.n_components / 100,
        f2.mean_curvature,
        f2.std_curvature,
        f2.orientation_entropy / np.log(16)
    ])
    
    # Weighted Euclidean distance
    weights = np.array([1.0, 0.5, 0.3, 0.3, 0.5])
    diff = (features1 - features2) * weights
    
    return float(np.sqrt(np.sum(diff**2)))


def extract_shell_topology(shell_mask: np.ndarray) -> Dict:
    """
    Extract topological features from shell structure.
    
    Returns:
        Dictionary with Betti numbers and other topology info.
    """
    if not np.any(shell_mask):
        return {
            'b0': 0,  # Connected components
            'b1': 0,  # Holes
            'euler': 0
        }
    
    # B0: Connected components
    labeled, n_components = ndimage.label(shell_mask)
    b0 = n_components
    
    # B1: Approximate holes by looking at background components
    background = ~shell_mask
    bg_labeled, n_bg = ndimage.label(background)
    # Holes = background components that don't touch border - 1 (exterior)
    b1 = max(0, n_bg - 1)
    
    # Euler characteristic: χ = B0 - B1
    euler = b0 - b1
    
    return {
        'b0': b0,
        'b1': b1,
        'euler': euler
    }


def shells_to_field(shell_mask: np.ndarray, blur_sigma: float = 1.0) -> ScalarField:
    """
    Convert shell mask to smooth scalar field.
    Useful for comparing shells in continuous space.
    """
    field = ScalarField.from_array(shell_mask.astype(np.float32))
    
    if blur_sigma > 0:
        field.data = ndimage.gaussian_filter(field.data, sigma=blur_sigma)
    
    return field
