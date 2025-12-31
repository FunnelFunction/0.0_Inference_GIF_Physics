"""
omega_tracker.py - Memory Accumulator (Ω^) for Inference Physics

The memory field Ω^ tracks where important dynamics have occurred.
It accumulates gradient history weighted by shell activity.

Mathematical Definition:
    Ω^_{n+1} = γ·Ω^_n + (1-γ)·ρq·|Φ|
    
Where:
    γ = decay rate (memory persistence)
    ρq = shell detection (where boundaries are)
    |Φ| = field magnitude (how active)

The memory tells us: "Where has the field been working?"

In ARC inference:
    Ω^ reveals which regions carry transformation information.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .scalar_field import ScalarField
from .shell_extractor import ShellExtractor


@dataclass
class MemoryStats:
    """Statistics about the memory field."""
    total_accumulated: float
    max_memory: float
    mean_memory: float
    active_fraction: float  # Fraction of field with significant memory
    hotspots: int          # Number of high-memory regions


class OmegaTracker:
    """
    Tracks accumulated memory field Ω^.
    
    Memory accumulates where:
        1. Shells form (boundaries are active)
        2. Field values are large (strong signal)
        3. Changes occur (dynamic regions)
    """
    
    def __init__(self, nx: int, ny: int, 
                 gamma: float = 0.95,
                 shell_weight: float = 1.0,
                 field_weight: float = 0.5,
                 change_weight: float = 0.3):
        """
        Args:
            gamma: Memory decay rate (higher = longer memory)
            shell_weight: Weight for shell activity
            field_weight: Weight for field magnitude
            change_weight: Weight for field changes
        """
        self.gamma = gamma
        self.shell_weight = shell_weight
        self.field_weight = field_weight
        self.change_weight = change_weight
        
        # Initialize memory field
        self.omega = ScalarField(nx, ny)
        self.omega.name = "Ω^"
        
        # Track previous field for change detection
        self.prev_field: Optional[np.ndarray] = None
        
        # Shell extractor
        self.shell_extractor = ShellExtractor(threshold_sigma=1.0)
        
        # History
        self.history: List[MemoryStats] = []
    
    def reset(self):
        """Reset memory to zero."""
        self.omega.data.fill(0)
        self.prev_field = None
        self.history = []
    
    def update(self, field: ScalarField, 
               shell_mask: Optional[np.ndarray] = None) -> MemoryStats:
        """
        Update memory based on current field state.
        
        Ω^_{n+1} = γ·Ω^_n + (1-γ)·(w_s·Σ + w_f·|Φ| + w_c·|ΔΦ|)
        """
        # Get shell mask if not provided
        if shell_mask is None:
            shell_mask = self.shell_extractor.extract(field)
        
        # Compute contribution terms
        shell_activity = shell_mask.astype(np.float32)
        field_magnitude = np.abs(field.data)
        
        if self.prev_field is not None:
            field_change = np.abs(field.data - self.prev_field)
        else:
            field_change = np.zeros_like(field.data)
        
        # Normalize to [0, 1]
        def normalize(x):
            max_val = np.max(x)
            return x / (max_val + 1e-8) if max_val > 0 else x
        
        shell_activity = normalize(shell_activity)
        field_magnitude = normalize(field_magnitude)
        field_change = normalize(field_change)
        
        # Weighted combination
        new_activity = (
            self.shell_weight * shell_activity +
            self.field_weight * field_magnitude +
            self.change_weight * field_change
        )
        
        # Normalize combined activity
        new_activity = normalize(new_activity)
        
        # Update memory with decay
        self.omega.data = (
            self.gamma * self.omega.data +
            (1 - self.gamma) * new_activity
        )
        
        # Store current field for next change detection
        self.prev_field = field.data.copy()
        
        # Compute stats
        stats = self.stats()
        self.history.append(stats)
        
        return stats
    
    def stats(self) -> MemoryStats:
        """Compute memory statistics."""
        data = self.omega.data
        
        total = float(np.sum(data))
        max_val = float(np.max(data))
        mean_val = float(np.mean(data))
        
        # Active = above 10% of max
        threshold = 0.1 * max_val
        active = np.sum(data > threshold) / data.size
        
        # Count hotspots (local maxima above threshold)
        from scipy import ndimage
        if max_val > 0:
            local_max = ndimage.maximum_filter(data, size=3) == data
            hotspots = int(np.sum(local_max & (data > threshold)))
        else:
            hotspots = 0
        
        return MemoryStats(
            total_accumulated=total,
            max_memory=max_val,
            mean_memory=mean_val,
            active_fraction=active,
            hotspots=hotspots
        )
    
    def get_averaged(self) -> ScalarField:
        """
        Get normalized memory field Ω^_avg.
        
        Used in collapse consistency computation.
        """
        result = self.omega.clone()
        max_val = np.max(result.data)
        if max_val > 0:
            result.data = result.data / max_val
        result.name = "Ω^_avg"
        return result
    
    def get_mask(self, threshold: float = 0.3) -> np.ndarray:
        """
        Get binary mask of high-memory regions.
        """
        max_val = np.max(self.omega.data)
        if max_val == 0:
            return np.zeros_like(self.omega.data, dtype=bool)
        
        normalized = self.omega.data / max_val
        return normalized > threshold
    
    def get_weighted_field(self, field: ScalarField) -> ScalarField:
        """
        Weight field by memory importance.
        
        Returns Φ·Ω^_avg
        """
        omega_avg = self.get_averaged()
        result = field.clone()
        result.data = result.data * omega_avg.data
        result.name = "Φ·Ω^"
        return result


class GradientTracker(OmegaTracker):
    """
    Specialized memory that tracks gradient directions.
    
    Instead of scalar memory, tracks the accumulated gradient field.
    """
    
    def __init__(self, nx: int, ny: int, gamma: float = 0.95):
        super().__init__(nx, ny, gamma)
        
        # Track gradient components
        self.omega_x = ScalarField(nx, ny)
        self.omega_y = ScalarField(nx, ny)
    
    def reset(self):
        super().reset()
        self.omega_x.data.fill(0)
        self.omega_y.data.fill(0)
    
    def update(self, field: ScalarField,
               shell_mask: Optional[np.ndarray] = None) -> MemoryStats:
        """Update gradient memory."""
        # Get gradients
        grad_x, grad_y = field.gradient()
        
        # Shell weighting
        if shell_mask is None:
            shell_mask = self.shell_extractor.extract(field)
        
        weight = shell_mask.astype(np.float32)
        
        # Update gradient memory with decay
        self.omega_x.data = (
            self.gamma * self.omega_x.data +
            (1 - self.gamma) * weight * grad_x.data
        )
        self.omega_y.data = (
            self.gamma * self.omega_y.data +
            (1 - self.gamma) * weight * grad_y.data
        )
        
        # Update magnitude memory (parent class)
        grad_mag = np.sqrt(grad_x.data**2 + grad_y.data**2)
        self.omega.data = (
            self.gamma * self.omega.data +
            (1 - self.gamma) * weight * grad_mag
        )
        
        self.prev_field = field.data.copy()
        stats = self.stats()
        self.history.append(stats)
        
        return stats
    
    def get_average_gradient(self) -> Tuple[ScalarField, ScalarField]:
        """Get averaged gradient field."""
        norm = np.sqrt(self.omega_x.data**2 + self.omega_y.data**2)
        max_norm = np.max(norm)
        
        avg_x = self.omega_x.clone()
        avg_y = self.omega_y.clone()
        
        if max_norm > 0:
            avg_x.data = avg_x.data / max_norm
            avg_y.data = avg_y.data / max_norm
        
        return avg_x, avg_y
    
    def gradient_consistency(self, field: ScalarField) -> float:
        """
        Measure how consistent field gradient is with accumulated memory.
        
        Returns value in [0, 1] where 1 = perfect consistency.
        """
        grad_x, grad_y = field.gradient()
        avg_x, avg_y = self.get_average_gradient()
        
        # Dot product of normalized gradients
        field_mag = np.sqrt(grad_x.data**2 + grad_y.data**2 + 1e-8)
        mem_mag = np.sqrt(avg_x.data**2 + avg_y.data**2 + 1e-8)
        
        dot = (grad_x.data * avg_x.data + grad_y.data * avg_y.data)
        cos_sim = dot / (field_mag * mem_mag + 1e-8)
        
        # Average over shell regions
        shell_mask = self.shell_extractor.extract(field)
        if np.sum(shell_mask) > 0:
            return float(np.mean(cos_sim[shell_mask]))
        else:
            return float(np.mean(cos_sim))
