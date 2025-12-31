"""
inference_solver.py - Main ARC-AGI Solver via Geometric Field Inference

This is the complete solver that:
    1. Loads ARC puzzle examples
    2. Encodes grids as scalar fields
    3. Computes transformation signatures
    4. Evolves fields under pure physics
    5. Infers the target transformation
    6. Applies to test input
    7. Decodes back to grid

The solver uses NO external training.
It infers the transformation from the structure of the examples.

Usage:
    solver = InferenceSolver()
    predictions = solver.solve(puzzle)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

from .scalar_field import ScalarField
from .field_dynamics import FieldDynamics, evolve_to_attractor
from .shell_extractor import ShellExtractor
from .omega_tracker import OmegaTracker
from .collapse_consistency import CollapseConsistency, ConsistencyScore
from .attractor_library import AttractorLibrary, discover_attractor
from .grid_codec import (
    GridCodec, encode_transformation, compose_transformations,
    apply_transformation, grid_accuracy
)


@dataclass
class SolverConfig:
    """Configuration for the inference solver."""
    # Encoding
    encoding_method: str = 'signed'
    pad_grids: bool = False
    pad_size: int = 32
    
    # Evolution
    max_evolution_steps: int = 50
    evolution_dt: float = 0.1
    evolution_tolerance: float = 1e-4
    evolution_operator: str = 'inference_dynamics'
    
    # Dynamics parameters
    dynamics_params: Optional[Dict] = None
    
    # Memory tracking
    memory_gamma: float = 0.95
    
    # Attractor matching
    use_attractor_library: bool = True
    max_composition: int = 2
    
    # Output
    discretize_output: bool = True
    
    def __post_init__(self):
        if self.dynamics_params is None:
            self.dynamics_params = {
                'eta': 0.05,
                'mu': 0.3,
                'lambda': 0.01
            }


@dataclass
class SolverResult:
    """Result of solving one test case."""
    predicted_grid: np.ndarray
    confidence: float
    evolution_steps: int
    matched_attractor: Optional[str]
    transformation_used: str
    solve_time: float
    debug_info: Dict


class InferenceSolver:
    """
    Main solver for ARC-AGI puzzles.
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        
        # Components
        self.codec = GridCodec(
            method=self.config.encoding_method,
            pad_to=self.config.pad_size if self.config.pad_grids else None
        )
        self.shell_extractor = ShellExtractor()
        
        # Attractor library
        self.library = AttractorLibrary()
        if self.config.use_attractor_library:
            self.library.build_base_attractors()
    
    def solve(self, puzzle: Dict) -> List[np.ndarray]:
        """
        Solve an ARC puzzle.
        
        Args:
            puzzle: Dictionary with 'train' and 'test' keys
                train: List of {'input': grid, 'output': grid}
                test: List of {'input': grid}
        
        Returns:
            List of predicted output grids for each test case
        """
        train_examples = puzzle['train']
        test_cases = puzzle['test']
        
        # Step 1: Analyze training examples
        transformation_info = self._analyze_examples(train_examples)
        
        # Step 2: Solve each test case
        predictions = []
        for test_case in test_cases:
            input_grid = np.array(test_case['input'])
            result = self._solve_test_case(input_grid, transformation_info)
            predictions.append(result.predicted_grid)
        
        return predictions
    
    def solve_with_results(self, puzzle: Dict) -> List[SolverResult]:
        """
        Solve with detailed results for each test case.
        """
        train_examples = puzzle['train']
        test_cases = puzzle['test']
        
        transformation_info = self._analyze_examples(train_examples)
        
        results = []
        for test_case in test_cases:
            input_grid = np.array(test_case['input'])
            result = self._solve_test_case(input_grid, transformation_info)
            results.append(result)
        
        return results
    
    def _analyze_examples(self, examples: List[Dict]) -> Dict:
        """
        Analyze training examples to understand the transformation.
        """
        info = {
            'transformations': [],
            'composite_transform': None,
            'matched_attractor': None,
            'example_fields': [],
            'size_change': False,
            'input_shapes': [],
            'output_shapes': []
        }
        
        # Compute transformation signatures
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            info['input_shapes'].append(input_grid.shape)
            info['output_shapes'].append(output_grid.shape)
            
            if input_grid.shape != output_grid.shape:
                info['size_change'] = True
            
            # Encode transformation
            delta = encode_transformation(input_grid, output_grid, self.codec)
            info['transformations'].append(delta)
            
            # Store encoded fields
            info['example_fields'].append({
                'input': self.codec.encode(input_grid),
                'output': self.codec.encode(output_grid),
                'delta': delta
            })
        
        # Compose transformations
        if info['transformations']:
            info['composite_transform'] = compose_transformations(
                info['transformations']
            )
        
        # Try to match to known attractor
        if len(examples) > 0 and self.config.use_attractor_library:
            first_in = self.codec.encode(np.array(examples[0]['input']))
            first_out = self.codec.encode(np.array(examples[0]['output']))
            
            attractor_name, score = discover_attractor(
                first_in, first_out, self.library, 
                max_composition=self.config.max_composition
            )
            
            if score < 0.1:  # Good match
                info['matched_attractor'] = attractor_name
        
        return info
    
    def _solve_test_case(self, input_grid: np.ndarray, 
                          transformation_info: Dict) -> SolverResult:
        """
        Solve a single test case.
        """
        start_time = time.time()
        debug_info = {}
        
        # Encode input
        input_field = self.codec.encode(input_grid)
        original_shape = input_grid.shape
        
        # Strategy 1: Use matched attractor if found
        if transformation_info['matched_attractor']:
            attractor_name = transformation_info['matched_attractor']
            
            # Handle composite attractors
            if ' → ' in attractor_name:
                names = attractor_name.split(' → ')
                result_field = input_field
                for name in names:
                    attractor = self.library.get(name)
                    if attractor:
                        result_field = attractor(result_field)
            else:
                attractor = self.library.get(attractor_name)
                if attractor:
                    result_field = attractor(input_field)
                else:
                    result_field = input_field
            
            predicted = self.codec.decode(result_field, original_shape)
            
            return SolverResult(
                predicted_grid=predicted,
                confidence=0.9,
                evolution_steps=0,
                matched_attractor=attractor_name,
                transformation_used='attractor',
                solve_time=time.time() - start_time,
                debug_info={'method': 'attractor_library'}
            )
        
        # Strategy 2: Apply composite transformation
        if transformation_info['composite_transform'] is not None:
            delta = transformation_info['composite_transform']
            
            # Apply transformation
            result_field = apply_transformation(input_field, delta)
            
            # Evolve to clean up
            omega_tracker = OmegaTracker(
                result_field.nx, result_field.ny,
                gamma=self.config.memory_gamma
            )
            
            result_field, history = evolve_to_attractor(
                result_field,
                operator=self.config.evolution_operator,
                max_steps=self.config.max_evolution_steps,
                dt=self.config.evolution_dt,
                tolerance=self.config.evolution_tolerance,
                params=self.config.dynamics_params
            )
            
            # Decode
            predicted = self.codec.decode(result_field, original_shape)
            
            return SolverResult(
                predicted_grid=predicted,
                confidence=0.7,
                evolution_steps=len(history),
                matched_attractor=None,
                transformation_used='composite_delta',
                solve_time=time.time() - start_time,
                debug_info={
                    'method': 'transformation_field',
                    'evolution_steps': len(history)
                }
            )
        
        # Strategy 3: Pure evolution (fallback)
        omega_tracker = OmegaTracker(
            input_field.nx, input_field.ny,
            gamma=self.config.memory_gamma
        )
        
        result_field, history = evolve_to_attractor(
            input_field.clone(),
            operator=self.config.evolution_operator,
            max_steps=self.config.max_evolution_steps,
            dt=self.config.evolution_dt,
            tolerance=self.config.evolution_tolerance,
            params=self.config.dynamics_params
        )
        
        predicted = self.codec.decode(result_field, original_shape)
        
        return SolverResult(
            predicted_grid=predicted,
            confidence=0.3,
            evolution_steps=len(history),
            matched_attractor=None,
            transformation_used='pure_evolution',
            solve_time=time.time() - start_time,
            debug_info={
                'method': 'pure_evolution',
                'evolution_steps': len(history)
            }
        )


def evaluate_solver(solver: InferenceSolver, 
                    puzzles: Dict[str, Dict],
                    verbose: bool = True) -> Dict:
    """
    Evaluate solver on a set of puzzles.
    
    Args:
        solver: The inference solver
        puzzles: Dictionary of puzzle_id -> puzzle
        verbose: Print progress
    
    Returns:
        Dictionary with accuracy and per-puzzle results
    """
    results = {
        'total_puzzles': 0,
        'total_test_cases': 0,
        'correct_predictions': 0,
        'per_puzzle': {}
    }
    
    for puzzle_id, puzzle in puzzles.items():
        results['total_puzzles'] += 1
        
        try:
            predictions = solver.solve(puzzle)
            test_cases = puzzle['test']
            
            puzzle_correct = 0
            puzzle_total = 0
            
            for i, (pred, test) in enumerate(zip(predictions, test_cases)):
                results['total_test_cases'] += 1
                puzzle_total += 1
                
                if 'output' in test:  # Has ground truth
                    target = np.array(test['output'])
                    if np.array_equal(pred, target):
                        results['correct_predictions'] += 1
                        puzzle_correct += 1
            
            results['per_puzzle'][puzzle_id] = {
                'correct': puzzle_correct,
                'total': puzzle_total,
                'accuracy': puzzle_correct / puzzle_total if puzzle_total > 0 else 0
            }
            
            if verbose:
                acc = puzzle_correct / puzzle_total if puzzle_total > 0 else 0
                print(f"  {puzzle_id}: {puzzle_correct}/{puzzle_total} ({acc:.1%})")
        
        except Exception as e:
            results['per_puzzle'][puzzle_id] = {
                'error': str(e)
            }
            if verbose:
                print(f"  {puzzle_id}: ERROR - {e}")
    
    results['accuracy'] = (
        results['correct_predictions'] / results['total_test_cases']
        if results['total_test_cases'] > 0 else 0
    )
    
    if verbose:
        print(f"\nOverall: {results['correct_predictions']}/{results['total_test_cases']} "
              f"({results['accuracy']:.1%})")
    
    return results
