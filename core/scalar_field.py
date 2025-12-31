"""
scalar_field.py - Core ScalarField class for field-based computation

The fundamental data structure for Inference GIF Physics.
A scalar field Φ: ℝ² → ℝ discretized on a grid.

Mathematical Foundation:
    Φ(x, y, t) ∈ ℝ for (x,y) ∈ Ω ⊂ ℝ²
    
This is the computational substrate - not weights, not symbols, just fields.
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class FieldMetrics:
    """Diagnostic metrics for a field state."""
    min_val: float
    max_val: float
    mean: float
    std: float
    energy: float  # Dirichlet energy ∫|∇Φ|²
    total_variation: float  # ∫|∇Φ|


class ScalarField:
    """
    A 2D scalar field with differential operators.
    
    Attributes:
        data: The field values as a 2D numpy array
        nx, ny: Grid dimensions
        dx, dy: Grid spacing (default 1.0)
    
    Mathematical Operations:
        - laplacian(): ∇²Φ
        - gradient(): (∂Φ/∂x, ∂Φ/∂y)
        - gradient_magnitude(): |∇Φ|
        - biharmonic(): ∇⁴Φ
    """
    
    def __init__(self, nx: int, ny: int, dx: float = 1.0, dy: float = 1.0):
        """Initialize empty field."""
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.data = np.zeros((ny, nx), dtype=np.float32)
        
        # Optional metadata
        self.pattern_wavelength: Optional[float] = None
        self.name: str = ""
    
    @classmethod
    def from_array(cls, arr: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> 'ScalarField':
        """Create field from numpy array."""
        ny, nx = arr.shape
        field = cls(nx, ny, dx, dy)
        field.data = arr.astype(np.float32)
        return field
    
    @classmethod
    def from_grid(cls, grid: np.ndarray, encoding: str = 'signed') -> 'ScalarField':
        """
        Create field from ARC grid (integers 0-9).
        
        Encodings:
            'signed': Φ = (grid - 4.5) / 4.5  → [-1, 1]
            'normalized': Φ = grid / 9  → [0, 1]
            'direct': Φ = grid  → [0, 9]
        """
        grid = np.array(grid, dtype=np.float32)
        ny, nx = grid.shape
        field = cls(nx, ny)
        
        if encoding == 'signed':
            field.data = (grid - 4.5) / 4.5
        elif encoding == 'normalized':
            field.data = grid / 9.0
        elif encoding == 'direct':
            field.data = grid
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        
        return field
    
    def to_grid(self, encoding: str = 'signed') -> np.ndarray:
        """
        Convert field back to ARC grid (integers 0-9).
        """
        if encoding == 'signed':
            scaled = self.data * 4.5 + 4.5
        elif encoding == 'normalized':
            scaled = self.data * 9.0
        elif encoding == 'direct':
            scaled = self.data
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        
        return np.clip(np.round(scaled), 0, 9).astype(np.int32)
    
    def clone(self) -> 'ScalarField':
        """Create a deep copy."""
        field = ScalarField(self.nx, self.ny, self.dx, self.dy)
        field.data = self.data.copy()
        field.pattern_wavelength = self.pattern_wavelength
        field.name = self.name
        return field
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DIFFERENTIAL OPERATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def laplacian(self) -> 'ScalarField':
        """
        Compute Laplacian ∇²Φ using 5-point stencil.
        
        ∇²Φ ≈ (Φ[i+1,j] + Φ[i-1,j] + Φ[i,j+1] + Φ[i,j-1] - 4Φ[i,j]) / h²
        
        Uses periodic boundary conditions.
        """
        result = ScalarField(self.nx, self.ny, self.dx, self.dy)
        
        # Roll for periodic boundaries
        left = np.roll(self.data, 1, axis=1)
        right = np.roll(self.data, -1, axis=1)
        up = np.roll(self.data, 1, axis=0)
        down = np.roll(self.data, -1, axis=0)
        
        h2 = self.dx * self.dy  # Assume dx ≈ dy
        result.data = (left + right + up + down - 4 * self.data) / h2
        
        return result
    
    def gradient(self) -> Tuple['ScalarField', 'ScalarField']:
        """
        Compute gradient (∂Φ/∂x, ∂Φ/∂y) using central differences.
        """
        grad_x = ScalarField(self.nx, self.ny, self.dx, self.dy)
        grad_y = ScalarField(self.nx, self.ny, self.dx, self.dy)
        
        # Central differences with periodic boundaries
        right = np.roll(self.data, -1, axis=1)
        left = np.roll(self.data, 1, axis=1)
        down = np.roll(self.data, -1, axis=0)
        up = np.roll(self.data, 1, axis=0)
        
        grad_x.data = (right - left) / (2 * self.dx)
        grad_y.data = (down - up) / (2 * self.dy)
        
        return grad_x, grad_y
    
    def gradient_magnitude(self) -> 'ScalarField':
        """
        Compute |∇Φ| = √((∂Φ/∂x)² + (∂Φ/∂y)²)
        """
        grad_x, grad_y = self.gradient()
        result = ScalarField(self.nx, self.ny, self.dx, self.dy)
        result.data = np.sqrt(grad_x.data**2 + grad_y.data**2)
        return result
    
    def biharmonic(self) -> 'ScalarField':
        """
        Compute biharmonic ∇⁴Φ = ∇²(∇²Φ)
        """
        return self.laplacian().laplacian()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIELD OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def sharpen(self, alpha: float = 0.2) -> 'ScalarField':
        """
        Laplacian sharpening: Φ_new = Φ - α·∇²Φ
        
        Boosts high frequencies (edges).
        """
        result = self.clone()
        lap = self.laplacian()
        result.data = self.data - alpha * lap.data
        return result
    
    def clamp(self, min_val: float = -1.5, max_val: float = 1.5) -> 'ScalarField':
        """Clamp values to range."""
        result = self.clone()
        result.data = np.clip(result.data, min_val, max_val)
        return result
    
    def normalize(self) -> 'ScalarField':
        """Normalize to [-1, 1]."""
        result = self.clone()
        min_v, max_v = result.data.min(), result.data.max()
        if max_v - min_v > 1e-8:
            result.data = 2 * (result.data - min_v) / (max_v - min_v) - 1
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def metrics(self) -> FieldMetrics:
        """Compute diagnostic metrics."""
        grad_mag = self.gradient_magnitude()
        
        return FieldMetrics(
            min_val=float(self.data.min()),
            max_val=float(self.data.max()),
            mean=float(self.data.mean()),
            std=float(self.data.std()),
            energy=float(np.sum(grad_mag.data**2) * self.dx * self.dy),
            total_variation=float(np.sum(grad_mag.data) * self.dx * self.dy)
        )
    
    def dirichlet_energy(self) -> float:
        """E[Φ] = ∫|∇Φ|² dx dy"""
        grad_mag = self.gradient_magnitude()
        return float(np.sum(grad_mag.data**2) * self.dx * self.dy)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOSS FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def l1_loss(self, target: 'ScalarField') -> float:
        """L1 = Σ|Φ - Φ*|"""
        return float(np.sum(np.abs(self.data - target.data)))
    
    def l2_loss(self, target: 'ScalarField') -> float:
        """L2 = √(Σ(Φ - Φ*)²)"""
        return float(np.sqrt(np.sum((self.data - target.data)**2)))
    
    def sign_match(self, target: 'ScalarField') -> float:
        """
        Percentage of pixels with matching sign.
        Useful for bistable convergence monitoring.
        """
        matches = np.sign(self.data) == np.sign(target.data)
        return float(np.mean(matches) * 100)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ARITHMETIC
    # ═══════════════════════════════════════════════════════════════════════════
    
    def __add__(self, other: Union['ScalarField', float]) -> 'ScalarField':
        result = self.clone()
        if isinstance(other, ScalarField):
            result.data = self.data + other.data
        else:
            result.data = self.data + other
        return result
    
    def __sub__(self, other: Union['ScalarField', float]) -> 'ScalarField':
        result = self.clone()
        if isinstance(other, ScalarField):
            result.data = self.data - other.data
        else:
            result.data = self.data - other
        return result
    
    def __mul__(self, other: Union['ScalarField', float]) -> 'ScalarField':
        result = self.clone()
        if isinstance(other, ScalarField):
            result.data = self.data * other.data
        else:
            result.data = self.data * other
        return result
    
    def __neg__(self) -> 'ScalarField':
        result = self.clone()
        result.data = -result.data
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REPRESENTATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def __repr__(self) -> str:
        m = self.metrics()
        return f"ScalarField({self.nx}×{self.ny}, Φ∈[{m.min_val:.3f}, {m.max_val:.3f}], E={m.energy:.2f})"


# ═══════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def noise_field(nx: int, ny: int, amplitude: float = 0.5, seed: Optional[int] = None) -> ScalarField:
    """Random noise field."""
    if seed is not None:
        np.random.seed(seed)
    field = ScalarField(nx, ny)
    field.data = amplitude * (2 * np.random.rand(ny, nx).astype(np.float32) - 1)
    field.name = f"noise(A={amplitude})"
    return field


def checkerboard_field(nx: int, ny: int, size: int = 20) -> ScalarField:
    """Checkerboard pattern."""
    field = ScalarField(nx, ny)
    field.pattern_wavelength = 2 * size
    
    for j in range(ny):
        for i in range(nx):
            xi = i // size
            yi = j // size
            field.data[j, i] = 1.0 if (xi + yi) % 2 == 0 else -1.0
    
    field.name = f"checkerboard(size={size})"
    return field


def disk_field(nx: int, ny: int, radius: float = 0.3, sharpness: float = 5.0) -> ScalarField:
    """Circular disk."""
    field = ScalarField(nx, ny)
    cx, cy = nx / 2, ny / 2
    r_pixels = radius * min(nx, ny)
    
    for j in range(ny):
        for i in range(nx):
            dist = np.sqrt((i - cx)**2 + (j - cy)**2)
            field.data[j, i] = -np.tanh(sharpness * (dist - r_pixels))
    
    field.name = f"disk(r={radius})"
    return field
