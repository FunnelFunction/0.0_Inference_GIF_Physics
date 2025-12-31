"""
grid_codec.py - ARC Grid ↔ Scalar Field Encoding/Decoding

Converts between ARC's discrete grid format (integers 0-9)
and our continuous scalar field representation.

ARC Format:
    - 2D arrays of integers 0-9
    - 0 is typically "background" (black)
    - 1-9 are colors
    - Grids range from 1×1 to 30×30

Field Encoding Strategies:
    1. Signed: Map [0,9] → [-1,1] (bistable compatible)
    2. Normalized: Map [0,9] → [0,1]
    3. Multi-channel: 10 binary channels (one per color)
    4. Spectral: FFT representation
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from .scalar_field import ScalarField


@dataclass
class GridEncoding:
    """Description of how a grid was encoded."""
    method: str
    n_channels: int
    value_range: Tuple[float, float]
    original_shape: Tuple[int, int]
    padded_shape: Optional[Tuple[int, int]]


class GridCodec:
    """
    Encode ARC grids to scalar fields and decode back.
    """
    
    # ARC color palette (approximate RGB for visualization)
    COLORS = {
        0: (0, 0, 0),       # Black (background)
        1: (0, 116, 217),   # Blue
        2: (255, 65, 54),   # Red
        3: (46, 204, 64),   # Green
        4: (255, 220, 0),   # Yellow
        5: (170, 170, 170), # Gray
        6: (240, 18, 190),  # Pink
        7: (255, 133, 27),  # Orange
        8: (127, 219, 255), # Cyan
        9: (135, 12, 37),   # Maroon
    }
    
    def __init__(self, method: str = 'signed', 
                 pad_to: Optional[int] = None):
        """
        Args:
            method: Encoding method ('signed', 'normalized', 'multichannel')
            pad_to: If set, pad grids to this size (square)
        """
        self.method = method
        self.pad_to = pad_to
    
    def encode(self, grid: Union[List, np.ndarray]) -> ScalarField:
        """
        Encode ARC grid to scalar field.
        """
        grid = np.array(grid, dtype=np.float32)
        original_shape = grid.shape
        
        # Pad if requested
        if self.pad_to is not None:
            grid = self._pad_grid(grid, self.pad_to)
        
        ny, nx = grid.shape
        field = ScalarField(nx, ny)
        
        if self.method == 'signed':
            # Map [0, 9] → [-1, 1]
            field.data = (grid - 4.5) / 4.5
        
        elif self.method == 'normalized':
            # Map [0, 9] → [0, 1]
            field.data = grid / 9.0
        
        elif self.method == 'centered':
            # Map [0, 9] → [-0.5, 0.5] (tighter range)
            field.data = (grid - 4.5) / 9.0
        
        elif self.method == 'binary':
            # Just 0 vs non-0
            field.data = (grid > 0).astype(np.float32) * 2 - 1
        
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")
        
        field.name = f"arc_grid({nx}×{ny})"
        return field
    
    def decode(self, field: ScalarField, 
               target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Decode scalar field back to ARC grid.
        """
        data = field.data
        
        # Unpad if needed
        if target_shape is not None:
            data = self._unpad(data, target_shape)
        
        if self.method == 'signed':
            # Map [-1, 1] → [0, 9]
            scaled = data * 4.5 + 4.5
        
        elif self.method == 'normalized':
            # Map [0, 1] → [0, 9]
            scaled = data * 9.0
        
        elif self.method == 'centered':
            # Map [-0.5, 0.5] → [0, 9]
            scaled = data * 9.0 + 4.5
        
        elif self.method == 'binary':
            # Map [-1, 1] → {0, 1}
            scaled = ((data + 1) / 2) * 9
        
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")
        
        # Round and clamp
        grid = np.clip(np.round(scaled), 0, 9).astype(np.int32)
        
        return grid
    
    def _pad_grid(self, grid: np.ndarray, target_size: int) -> np.ndarray:
        """Pad grid to target size with zeros (background)."""
        ny, nx = grid.shape
        if ny >= target_size and nx >= target_size:
            return grid
        
        padded = np.zeros((target_size, target_size), dtype=grid.dtype)
        # Center the grid
        y_offset = (target_size - ny) // 2
        x_offset = (target_size - nx) // 2
        padded[y_offset:y_offset+ny, x_offset:x_offset+nx] = grid
        
        return padded
    
    def _unpad(self, data: np.ndarray, 
               target_shape: Tuple[int, int]) -> np.ndarray:
        """Remove padding to restore original shape."""
        current_shape = data.shape
        target_ny, target_nx = target_shape
        
        # Calculate offsets
        y_offset = (current_shape[0] - target_ny) // 2
        x_offset = (current_shape[1] - target_nx) // 2
        
        return data[y_offset:y_offset+target_ny, x_offset:x_offset+target_nx]


class MultiChannelCodec:
    """
    Encode ARC grids as multi-channel fields.
    
    Each color gets its own binary channel.
    Good for preserving color identity.
    """
    
    def __init__(self, n_colors: int = 10):
        self.n_colors = n_colors
    
    def encode(self, grid: Union[List, np.ndarray]) -> List[ScalarField]:
        """
        Encode to list of binary fields (one per color).
        """
        grid = np.array(grid, dtype=np.int32)
        ny, nx = grid.shape
        
        channels = []
        for c in range(self.n_colors):
            field = ScalarField(nx, ny)
            field.data = (grid == c).astype(np.float32) * 2 - 1
            field.name = f"channel_{c}"
            channels.append(field)
        
        return channels
    
    def decode(self, channels: List[ScalarField]) -> np.ndarray:
        """
        Decode from multi-channel to grid.
        
        Each pixel gets the color of the channel with highest value.
        """
        if not channels:
            raise ValueError("No channels provided")
        
        ny, nx = channels[0].ny, channels[0].nx
        
        # Stack channels and find argmax
        stacked = np.stack([ch.data for ch in channels], axis=0)
        grid = np.argmax(stacked, axis=0).astype(np.int32)
        
        return grid


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMATION ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_transformation(input_grid: np.ndarray, 
                          output_grid: np.ndarray,
                          codec: Optional[GridCodec] = None) -> ScalarField:
    """
    Encode a transformation as a field difference.
    
    ΔΦ = Lift(B) - Lift(A)
    """
    codec = codec or GridCodec(method='signed')
    
    field_in = codec.encode(input_grid)
    field_out = codec.encode(output_grid)
    
    # Handle size mismatch by padding to larger
    if field_in.data.shape != field_out.data.shape:
        max_ny = max(field_in.ny, field_out.ny)
        max_nx = max(field_in.nx, field_out.nx)
        
        field_in = _resize_field(field_in, max_nx, max_ny)
        field_out = _resize_field(field_out, max_nx, max_ny)
    
    delta = field_out - field_in
    delta.name = "ΔΦ"
    
    return delta


def _resize_field(field: ScalarField, nx: int, ny: int) -> ScalarField:
    """Resize field by padding with zeros."""
    if field.nx == nx and field.ny == ny:
        return field
    
    result = ScalarField(nx, ny, field.dx, field.dy)
    
    # Center the original
    y_offset = (ny - field.ny) // 2
    x_offset = (nx - field.nx) // 2
    
    result.data[y_offset:y_offset+field.ny, 
                x_offset:x_offset+field.nx] = field.data
    
    return result


def compose_transformations(deltas: List[ScalarField]) -> ScalarField:
    """
    Compose multiple transformation fields into one.
    
    Options:
        - Average: Simple mean
        - Consensus: Keep only common elements
        - Weighted: Weight by consistency
    """
    if not deltas:
        raise ValueError("No transformations to compose")
    
    if len(deltas) == 1:
        return deltas[0].clone()
    
    # Ensure all same size
    max_ny = max(d.ny for d in deltas)
    max_nx = max(d.nx for d in deltas)
    
    resized = [_resize_field(d, max_nx, max_ny) for d in deltas]
    
    # Average
    result = ScalarField(max_nx, max_ny)
    result.data = np.mean([d.data for d in resized], axis=0)
    result.name = "ΔΦ_composed"
    
    return result


def apply_transformation(field: ScalarField, 
                         delta: ScalarField) -> ScalarField:
    """
    Apply transformation field to input.
    
    Φ_out = Φ_in + ΔΦ
    """
    if field.data.shape != delta.data.shape:
        delta = _resize_field(delta, field.nx, field.ny)
    
    result = field + delta
    result.name = f"{field.name}_transformed"
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def grid_to_image(grid: np.ndarray) -> np.ndarray:
    """
    Convert ARC grid to RGB image for visualization.
    """
    ny, nx = grid.shape
    image = np.zeros((ny, nx, 3), dtype=np.uint8)
    
    for c, rgb in GridCodec.COLORS.items():
        mask = grid == c
        image[mask] = rgb
    
    return image


def grids_equal(grid1: np.ndarray, grid2: np.ndarray) -> bool:
    """Check if two grids are identical."""
    return np.array_equal(grid1, grid2)


def grid_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Compute pixel-wise accuracy between grids.
    """
    if predicted.shape != target.shape:
        return 0.0
    
    return float(np.mean(predicted == target))
