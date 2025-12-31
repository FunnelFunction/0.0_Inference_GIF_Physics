"""
Inference GIF Physics - Core Module

Solve ARC-AGI via geometric field inference.
"""

from .scalar_field import ScalarField, noise_field, checkerboard_field, disk_field
from .field_dynamics import FieldDynamics, evolve_to_attractor
from .shell_extractor import ShellExtractor, shell_distance, ShellFeatures
from .omega_tracker import OmegaTracker, GradientTracker
from .collapse_consistency import CollapseConsistency, ConsistencyScore
from .attractor_library import AttractorLibrary, Attractor, TransformType
from .grid_codec import GridCodec, encode_transformation, compose_transformations
from .inference_solver import InferenceSolver, SolverConfig, SolverResult

__all__ = [
    # Scalar Field
    'ScalarField',
    'noise_field',
    'checkerboard_field', 
    'disk_field',
    
    # Dynamics
    'FieldDynamics',
    'evolve_to_attractor',
    
    # Shell Extraction
    'ShellExtractor',
    'shell_distance',
    'ShellFeatures',
    
    # Memory
    'OmegaTracker',
    'GradientTracker',
    
    # Consistency
    'CollapseConsistency',
    'ConsistencyScore',
    
    # Attractors
    'AttractorLibrary',
    'Attractor',
    'TransformType',
    
    # Grid Encoding
    'GridCodec',
    'encode_transformation',
    'compose_transformations',
    
    # Solver
    'InferenceSolver',
    'SolverConfig',
    'SolverResult'
]
