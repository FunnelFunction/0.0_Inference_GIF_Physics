"""
attractor_library.py - Library of Transformation Attractors (Λ)

This is the set of known transformation patterns that the system
can match observed dynamics against.

Each attractor represents a TYPE of transformation:
    - Rotations (90°, 180°, 270°)
    - Reflections (horizontal, vertical)
    - Scaling
    - Color swaps
    - Pattern repetitions
    - etc.

The attractor is not a single field, but a TEMPLATE that describes
how fields transform under that operation.

For ARC:
    We build attractors from common puzzle transformations.
    When we observe a field evolving, we ask: which attractor
    best explains the observed dynamics?
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from .scalar_field import ScalarField
from .shell_extractor import ShellExtractor, ShellFeatures
from .collapse_consistency import ConsistencyScore, CollapseConsistency


class TransformType(Enum):
    """Categories of transformations."""
    IDENTITY = "identity"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    TRANSLATION = "translation"
    SCALING = "scaling"
    COLOR = "color"
    TILING = "tiling"
    COMPOSITION = "composition"


@dataclass
class Attractor:
    """
    A transformation attractor in the library.
    """
    name: str
    transform_type: TransformType
    apply: Callable[[ScalarField], ScalarField]  # The actual transform
    signature: Optional[np.ndarray] = None       # Characteristic signature
    description: str = ""
    
    def __call__(self, field: ScalarField) -> ScalarField:
        """Apply the transformation."""
        return self.apply(field)


class AttractorLibrary:
    """
    Library of transformation attractors.
    
    Use:
        library = AttractorLibrary()
        library.build_base_attractors()
        
        # Find best match
        best = library.match(omega, sigma, rho_q, observed_field)
    """
    
    def __init__(self):
        self.attractors: Dict[str, Attractor] = {}
        self.shell_extractor = ShellExtractor()
        self.consistency = CollapseConsistency()
    
    def add(self, attractor: Attractor):
        """Add attractor to library."""
        self.attractors[attractor.name] = attractor
    
    def get(self, name: str) -> Optional[Attractor]:
        """Get attractor by name."""
        return self.attractors.get(name)
    
    def list(self) -> List[str]:
        """List all attractor names."""
        return list(self.attractors.keys())
    
    def build_base_attractors(self):
        """
        Build library of common ARC transformations.
        """
        # ═══════════════════════════════════════════════════════════════════════
        # IDENTITY
        # ═══════════════════════════════════════════════════════════════════════
        
        self.add(Attractor(
            name="identity",
            transform_type=TransformType.IDENTITY,
            apply=lambda f: f.clone(),
            description="No change"
        ))
        
        # ═══════════════════════════════════════════════════════════════════════
        # ROTATIONS
        # ═══════════════════════════════════════════════════════════════════════
        
        def rotate_90(field: ScalarField) -> ScalarField:
            result = field.clone()
            result.data = np.rot90(field.data, k=1)
            result.nx, result.ny = result.ny, result.nx
            return result
        
        def rotate_180(field: ScalarField) -> ScalarField:
            result = field.clone()
            result.data = np.rot90(field.data, k=2)
            return result
        
        def rotate_270(field: ScalarField) -> ScalarField:
            result = field.clone()
            result.data = np.rot90(field.data, k=3)
            result.nx, result.ny = result.ny, result.nx
            return result
        
        self.add(Attractor("rotate_90", TransformType.ROTATION, rotate_90, 
                          description="90° counterclockwise"))
        self.add(Attractor("rotate_180", TransformType.ROTATION, rotate_180,
                          description="180° rotation"))
        self.add(Attractor("rotate_270", TransformType.ROTATION, rotate_270,
                          description="270° counterclockwise"))
        
        # ═══════════════════════════════════════════════════════════════════════
        # REFLECTIONS
        # ═══════════════════════════════════════════════════════════════════════
        
        def flip_horizontal(field: ScalarField) -> ScalarField:
            result = field.clone()
            result.data = np.fliplr(field.data)
            return result
        
        def flip_vertical(field: ScalarField) -> ScalarField:
            result = field.clone()
            result.data = np.flipud(field.data)
            return result
        
        def flip_diagonal(field: ScalarField) -> ScalarField:
            result = field.clone()
            result.data = field.data.T
            result.nx, result.ny = result.ny, result.nx
            return result
        
        self.add(Attractor("flip_h", TransformType.REFLECTION, flip_horizontal,
                          description="Horizontal flip (left-right)"))
        self.add(Attractor("flip_v", TransformType.REFLECTION, flip_vertical,
                          description="Vertical flip (up-down)"))
        self.add(Attractor("flip_diag", TransformType.REFLECTION, flip_diagonal,
                          description="Diagonal flip (transpose)"))
        
        # ═══════════════════════════════════════════════════════════════════════
        # INVERSIONS
        # ═══════════════════════════════════════════════════════════════════════
        
        def invert(field: ScalarField) -> ScalarField:
            result = field.clone()
            result.data = -field.data
            return result
        
        def complement(field: ScalarField) -> ScalarField:
            """Complement in [0,1] space."""
            result = field.clone()
            # Map [-1,1] to [0,1], complement, map back
            normalized = (field.data + 1) / 2
            complemented = 1 - normalized
            result.data = complemented * 2 - 1
            return result
        
        self.add(Attractor("invert", TransformType.COLOR, invert,
                          description="Negate field values"))
        self.add(Attractor("complement", TransformType.COLOR, complement,
                          description="Color complement"))
        
        # ═══════════════════════════════════════════════════════════════════════
        # TILING
        # ═══════════════════════════════════════════════════════════════════════
        
        def tile_2x2(field: ScalarField) -> ScalarField:
            """Tile field into 2x2 grid."""
            result = ScalarField(field.nx * 2, field.ny * 2)
            for dy in [0, field.ny]:
                for dx in [0, field.nx]:
                    result.data[dy:dy+field.ny, dx:dx+field.nx] = field.data
            return result
        
        self.add(Attractor("tile_2x2", TransformType.TILING, tile_2x2,
                          description="Tile into 2x2 grid"))
        
        # ═══════════════════════════════════════════════════════════════════════
        # SCALING (Simple)
        # ═══════════════════════════════════════════════════════════════════════
        
        def scale_2x(field: ScalarField) -> ScalarField:
            """Double the resolution (nearest neighbor)."""
            result = ScalarField(field.nx * 2, field.ny * 2)
            result.data = np.repeat(np.repeat(field.data, 2, axis=0), 2, axis=1)
            return result
        
        self.add(Attractor("scale_2x", TransformType.SCALING, scale_2x,
                          description="Double resolution"))
    
    def match(self, omega: ScalarField, sigma: np.ndarray, 
              rho_q: ScalarField, observed_field: ScalarField,
              input_field: Optional[ScalarField] = None) -> Tuple[str, ConsistencyScore]:
        """
        Find best matching attractor.
        
        Args:
            omega: Accumulated memory
            sigma: Shell mask
            rho_q: Curvature field
            observed_field: The field we evolved to
            input_field: Optional original input (for transform comparison)
        
        Returns:
            (best_attractor_name, score)
        """
        scores = {}
        
        for name, attractor in self.attractors.items():
            if input_field is not None:
                # Transform input and compare to observed
                try:
                    transformed = attractor(input_field)
                    
                    # Resize if needed
                    if transformed.data.shape != observed_field.data.shape:
                        continue  # Skip mismatched sizes for now
                    
                    score = self.consistency.compute(
                        omega, sigma, rho_q, transformed, observed_field
                    )
                    scores[name] = score
                except Exception:
                    continue
            else:
                # Just check consistency with attractor signature
                # This is less accurate but doesn't require input
                pass
        
        if not scores:
            return "identity", ConsistencyScore(
                total=float('inf'), gradient_term=0, shell_term=0,
                curvature_term=0, field_term=0, weights_used={}
            )
        
        best_name = min(scores, key=lambda k: scores[k].total)
        return best_name, scores[best_name]
    
    def match_by_transform(self, input_field: ScalarField, 
                           output_field: ScalarField) -> Tuple[str, float]:
        """
        Find which attractor transforms input to output.
        
        Simpler method: directly compare T(input) to output.
        """
        scores = {}
        
        for name, attractor in self.attractors.items():
            try:
                transformed = attractor(input_field)
                
                if transformed.data.shape != output_field.data.shape:
                    scores[name] = float('inf')
                    continue
                
                # Direct L2 distance
                diff = transformed.data - output_field.data
                scores[name] = float(np.mean(diff**2))
            except Exception:
                scores[name] = float('inf')
        
        best_name = min(scores, key=scores.get)
        return best_name, scores[best_name]


class CompositeAttractor:
    """
    Compose multiple attractors into one.
    
    For complex transformations like "rotate then flip".
    """
    
    def __init__(self, library: AttractorLibrary, names: List[str]):
        self.library = library
        self.names = names
        self.attractors = [library.get(n) for n in names]
        
        if any(a is None for a in self.attractors):
            missing = [n for n, a in zip(names, self.attractors) if a is None]
            raise ValueError(f"Unknown attractors: {missing}")
    
    def __call__(self, field: ScalarField) -> ScalarField:
        """Apply all transforms in sequence."""
        result = field
        for attractor in self.attractors:
            result = attractor(result)
        return result
    
    @property
    def name(self) -> str:
        return " → ".join(self.names)


def discover_attractor(input_field: ScalarField, 
                       output_field: ScalarField,
                       library: AttractorLibrary,
                       max_composition: int = 2) -> Tuple[str, float]:
    """
    Discover which attractor (or composition) transforms input to output.
    
    Tries:
        1. Single attractors
        2. Compositions up to max_composition length
    """
    # Try singles
    best_name, best_score = library.match_by_transform(input_field, output_field)
    
    if best_score < 1e-6:
        return best_name, best_score
    
    # Try pairs
    if max_composition >= 2:
        for name1 in library.list():
            for name2 in library.list():
                try:
                    composite = CompositeAttractor(library, [name1, name2])
                    transformed = composite(input_field)
                    
                    if transformed.data.shape != output_field.data.shape:
                        continue
                    
                    diff = transformed.data - output_field.data
                    score = float(np.mean(diff**2))
                    
                    if score < best_score:
                        best_score = score
                        best_name = composite.name
                except Exception:
                    continue
    
    return best_name, best_score
