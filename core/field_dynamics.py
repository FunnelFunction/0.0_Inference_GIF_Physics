"""
field_dynamics.py - PDE Evolution Operators for Scalar Fields

The physics engine for Inference GIF Physics.
Contains evolution operators that drive fields toward their natural attractors.

Key Insight: We evolve WITHOUT a known target.
The attractor emerges from the field's own structure.

Mathematical Foundation:
    ∂Φ/∂t = F[Φ, ∇Φ, ∇²Φ, ∇⁴Φ]
    
    Different F give different attractor landscapes.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from .scalar_field import ScalarField


@dataclass
class EvolutionResult:
    """Result of one evolution step."""
    operator: str
    dt_actual: float
    substeps: int
    field_range: Tuple[float, float]
    energy: float
    max_change: float
    extra: Dict


class FieldDynamics:
    """
    Collection of PDE evolution operators.
    
    All operators evolve the field IN PLACE and return diagnostics.
    """
    
    @staticmethod
    def diffusion(field: ScalarField, dt: float, 
                  eta: float = 0.1) -> EvolutionResult:
        """
        Heat equation: ∂Φ/∂t = η∇²Φ
        
        Smooths the field. Dissipates high frequencies.
        Attractor: uniform field (boring, but stable).
        """
        lap = field.laplacian()
        
        old_data = field.data.copy()
        field.data = field.data + dt * eta * lap.data
        
        max_change = float(np.max(np.abs(field.data - old_data)))
        
        return EvolutionResult(
            operator='diffusion',
            dt_actual=dt,
            substeps=1,
            field_range=(float(field.data.min()), float(field.data.max())),
            energy=field.dirichlet_energy(),
            max_change=max_change,
            extra={'eta': eta}
        )
    
    @staticmethod
    def allen_cahn(field: ScalarField, dt: float,
                   eta: float = 0.1, mu: float = 1.0) -> EvolutionResult:
        """
        Allen-Cahn equation: ∂Φ/∂t = η∇²Φ + μΦ(1-Φ²)
        
        Bistable dynamics. Field is pushed toward ±1.
        Attractor: domains of ±1 separated by thin walls.
        """
        lap = field.laplacian()
        
        # Bistable reaction term
        reaction = mu * field.data * (1 - field.data**2)
        
        old_data = field.data.copy()
        field.data = field.data + dt * (eta * lap.data + reaction)
        
        # Stability clamp
        field.data = np.clip(field.data, -1.5, 1.5)
        
        max_change = float(np.max(np.abs(field.data - old_data)))
        
        return EvolutionResult(
            operator='allen_cahn',
            dt_actual=dt,
            substeps=1,
            field_range=(float(field.data.min()), float(field.data.max())),
            energy=field.dirichlet_energy(),
            max_change=max_change,
            extra={'eta': eta, 'mu': mu}
        )
    
    @staticmethod
    def swift_hohenberg(field: ScalarField, dt: float,
                        r: float = 0.3, g: float = 1.0,
                        k0: float = 1.0) -> EvolutionResult:
        """
        Swift-Hohenberg equation: ∂Φ/∂t = rΦ - (k₀² + ∇²)²Φ - gΦ³
        
        Pattern-forming dynamics with PREFERRED WAVELENGTH λ = 2π/k₀.
        
        Attractor: periodic patterns (stripes, hexagons, checkerboards).
        
        This is the KEY operator for ARC - it forms structured patterns.
        """
        k0_sq = k0 ** 2
        k0_4 = k0_sq ** 2
        
        lap = field.laplacian()
        bilap = lap.laplacian()
        
        # (k₀² + ∇²)²Φ = k₀⁴Φ + 2k₀²∇²Φ + ∇⁴Φ
        linear_op = k0_4 * field.data + 2 * k0_sq * lap.data + bilap.data
        
        # Cubic saturation
        reaction = field.data * (r - g * field.data**2)
        
        # Compute stable timestep
        dx4 = field.dx ** 4
        dt_safe = min(dx4 / 16, 0.5 / max(r, g, k0_4 + 0.1), 0.05)
        
        n_substeps = max(1, int(np.ceil(dt / dt_safe)))
        dt_sub = dt / n_substeps
        
        old_data = field.data.copy()
        
        for _ in range(n_substeps):
            lap = field.laplacian()
            bilap = lap.laplacian()
            linear_op = k0_4 * field.data + 2 * k0_sq * lap.data + bilap.data
            reaction = field.data * (r - g * field.data**2)
            
            field.data = field.data + dt_sub * (reaction - linear_op)
            
            # Stability clamp
            clamp = np.sqrt(max(r, 0.1) / g) * 1.5
            field.data = np.clip(field.data, -clamp, clamp)
        
        max_change = float(np.max(np.abs(field.data - old_data)))
        
        return EvolutionResult(
            operator='swift_hohenberg',
            dt_actual=dt,
            substeps=n_substeps,
            field_range=(float(field.data.min()), float(field.data.max())),
            energy=field.dirichlet_energy(),
            max_change=max_change,
            extra={'r': r, 'g': g, 'k0': k0, 'wavelength': 2 * np.pi / k0}
        )
    
    @staticmethod
    def css(field: ScalarField, dt: float,
            eta: float = 0.1, lam: float = 0.05, mu: float = 0.5,
            alpha: float = 0.0) -> EvolutionResult:
        """
        Collapse-Stabilize-Symbolize (CSS) dynamics:
        
        ∂Φ/∂t = η∇²Φ - λ|∇Φ|² + μΦ(1-Φ²) + α·δ_drift
        
        Components:
            η∇²Φ: Diffusion (smoothing)
            -λ|∇Φ|²: Gradient suppression (sharpens boundaries)
            μΦ(1-Φ²): Bistable reaction (push to ±1)
            α·δ_drift: Optional drift toward memory
        
        Attractor: clean domains with sharp boundaries.
        """
        lap = field.laplacian()
        grad_mag = field.gradient_magnitude()
        
        # CSS terms
        diffusion = eta * lap.data
        grad_suppress = -lam * grad_mag.data**2
        bistable = mu * field.data * (1 - field.data**2)
        
        old_data = field.data.copy()
        field.data = field.data + dt * (diffusion + grad_suppress + bistable)
        
        # Stability clamp
        field.data = np.clip(field.data, -1.5, 1.5)
        
        max_change = float(np.max(np.abs(field.data - old_data)))
        
        return EvolutionResult(
            operator='css',
            dt_actual=dt,
            substeps=1,
            field_range=(float(field.data.min()), float(field.data.max())),
            energy=field.dirichlet_energy(),
            max_change=max_change,
            extra={'eta': eta, 'lambda': lam, 'mu': mu}
        )
    
    @staticmethod
    def inference_dynamics(field: ScalarField, dt: float,
                          params: Optional[Dict] = None) -> EvolutionResult:
        """
        Combined dynamics optimized for ARC inference:
        
        ∂Φ/∂t = η∇²Φ + μΦ(1-Φ²) - λ∇⁴Φ
        
        Balances:
            - Smoothing (coherent regions)
            - Bistability (discrete values)
            - Pattern formation (structure preservation)
        """
        params = params or {}
        eta = params.get('eta', 0.05)
        mu = params.get('mu', 0.3)
        lam = params.get('lambda', 0.01)
        
        lap = field.laplacian()
        bilap = lap.laplacian()
        
        diffusion = eta * lap.data
        bistable = mu * field.data * (1 - field.data**2)
        hyperdiffusion = -lam * bilap.data
        
        old_data = field.data.copy()
        field.data = field.data + dt * (diffusion + bistable + hyperdiffusion)
        
        # Clamp to valid range
        field.data = np.clip(field.data, -1.2, 1.2)
        
        max_change = float(np.max(np.abs(field.data - old_data)))
        
        return EvolutionResult(
            operator='inference_dynamics',
            dt_actual=dt,
            substeps=1,
            field_range=(float(field.data.min()), float(field.data.max())),
            energy=field.dirichlet_energy(),
            max_change=max_change,
            extra={'eta': eta, 'mu': mu, 'lambda': lam}
        )


def evolve_to_attractor(field: ScalarField, 
                        operator: str = 'inference_dynamics',
                        max_steps: int = 100,
                        dt: float = 0.1,
                        tolerance: float = 1e-4,
                        params: Optional[Dict] = None) -> Tuple[ScalarField, list]:
    """
    Evolve field until it reaches a stable attractor.
    
    Returns:
        field: The evolved field (modified in place)
        history: List of EvolutionResult for each step
    """
    params = params or {}
    history = []
    
    # Get operator function
    ops = {
        'diffusion': FieldDynamics.diffusion,
        'allen_cahn': FieldDynamics.allen_cahn,
        'swift_hohenberg': FieldDynamics.swift_hohenberg,
        'css': FieldDynamics.css,
        'inference_dynamics': FieldDynamics.inference_dynamics
    }
    
    if operator not in ops:
        raise ValueError(f"Unknown operator: {operator}")
    
    evolve_fn = ops[operator]
    
    for step in range(max_steps):
        if operator == 'swift_hohenberg':
            result = evolve_fn(field, dt, 
                             r=params.get('r', 0.3),
                             g=params.get('g', 1.0),
                             k0=params.get('k0', 1.0))
        elif operator == 'css':
            result = evolve_fn(field, dt,
                             eta=params.get('eta', 0.1),
                             lam=params.get('lambda', 0.05),
                             mu=params.get('mu', 0.5))
        elif operator == 'allen_cahn':
            result = evolve_fn(field, dt,
                             eta=params.get('eta', 0.1),
                             mu=params.get('mu', 1.0))
        elif operator == 'inference_dynamics':
            result = evolve_fn(field, dt, params)
        else:
            result = evolve_fn(field, dt)
        
        history.append(result)
        
        # Check convergence
        if result.max_change < tolerance:
            break
    
    return field, history


def discretize_field(field: ScalarField, n_levels: int = 10) -> ScalarField:
    """
    Discretize continuous field to n discrete levels.
    Useful for converting back to ARC grid.
    """
    result = field.clone()
    
    # Map [-1, 1] to [0, n_levels-1]
    normalized = (result.data + 1) / 2  # [0, 1]
    discrete = np.round(normalized * (n_levels - 1))
    
    # Map back to [-1, 1]
    result.data = (discrete / (n_levels - 1)) * 2 - 1
    
    return result
