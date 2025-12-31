"""
collapse_consistency.py - The Core Inference Function C

This is the heart of inference physics:
    Φ* = argmin_{Φ̂ ∈ Λ} C(Ω^, Σ, ρq | Φ̂)

The collapse consistency function C measures how well a candidate
target Φ̂ explains the observed field dynamics (memory, shells, curvature).

Lower C = better match = more likely to be the true Φ*.

Mathematical Definition:
    C(Ω^, Σ, ρq | Φ̂) = α||∇Φ̂ - Ω^_avg||² + β·d(Σ, Σ̂) + γ||∇²Φ̂ - ρq||²

Components:
    Term 1: Does Φ̂ have gradients where we saw activity?
    Term 2: Does Φ̂ have shells where we saw shells?
    Term 3: Does Φ̂ have curvature where we saw curvature?
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .scalar_field import ScalarField
from .shell_extractor import ShellExtractor, shell_distance, ShellFeatures
from .omega_tracker import OmegaTracker


@dataclass
class ConsistencyScore:
    """Breakdown of collapse consistency score."""
    total: float
    gradient_term: float
    shell_term: float
    curvature_term: float
    field_term: float
    weights_used: Dict[str, float]


class CollapseConsistency:
    """
    Compute collapse consistency between observed dynamics and candidate target.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,    # Gradient weight
                 beta: float = 1.0,     # Shell weight  
                 gamma: float = 0.5,    # Curvature weight
                 delta: float = 0.3,    # Field value weight
                 shell_metric: str = 'iou'):
        """
        Args:
            alpha: Weight for gradient consistency
            beta: Weight for shell matching
            gamma: Weight for curvature consistency
            delta: Weight for direct field matching
            shell_metric: Distance metric for shells ('iou', 'dice', 'hausdorff')
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.shell_metric = shell_metric
        
        self.shell_extractor = ShellExtractor()
    
    def compute(self, 
                omega: ScalarField,       # Accumulated memory Ω^
                sigma: np.ndarray,        # Observed shell mask Σ
                rho_q: ScalarField,       # Observed curvature field ρq
                phi_hat: ScalarField,     # Candidate target Φ̂
                observed_field: Optional[ScalarField] = None  # Final observed field
               ) -> ConsistencyScore:
        """
        Compute collapse consistency score.
        
        C(Ω^, Σ, ρq | Φ̂) = α·C_grad + β·C_shell + γ·C_curv + δ·C_field
        """
        weights = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta
        }
        
        # Term 1: Gradient consistency
        # Does Φ̂ have high gradients where memory accumulated?
        grad_phi_hat = phi_hat.gradient_magnitude()
        
        # Normalize both for comparison
        omega_norm = omega.data / (np.max(omega.data) + 1e-8)
        grad_norm = grad_phi_hat.data / (np.max(grad_phi_hat.data) + 1e-8)
        
        gradient_term = float(np.mean((grad_norm - omega_norm)**2))
        
        # Term 2: Shell matching
        # Does Φ̂ have shells where we observed shells?
        sigma_hat = self.shell_extractor.extract(phi_hat)
        shell_term = shell_distance(sigma, sigma_hat, metric=self.shell_metric)
        
        # Term 3: Curvature consistency
        # Does Φ̂ have curvature structure matching ρq?
        lap_phi_hat = phi_hat.laplacian()
        
        rho_q_norm = rho_q.data / (np.max(np.abs(rho_q.data)) + 1e-8)
        lap_norm = lap_phi_hat.data / (np.max(np.abs(lap_phi_hat.data)) + 1e-8)
        
        curvature_term = float(np.mean((lap_norm - rho_q_norm)**2))
        
        # Term 4: Direct field matching (if observed field available)
        if observed_field is not None:
            phi_norm = observed_field.data / (np.max(np.abs(observed_field.data)) + 1e-8)
            phi_hat_norm = phi_hat.data / (np.max(np.abs(phi_hat.data)) + 1e-8)
            field_term = float(np.mean((phi_norm - phi_hat_norm)**2))
        else:
            field_term = 0.0
        
        # Weighted sum
        total = (
            self.alpha * gradient_term +
            self.beta * shell_term +
            self.gamma * curvature_term +
            self.delta * field_term
        )
        
        return ConsistencyScore(
            total=total,
            gradient_term=gradient_term,
            shell_term=shell_term,
            curvature_term=curvature_term,
            field_term=field_term,
            weights_used=weights
        )
    
    def compute_from_tracker(self,
                             tracker: OmegaTracker,
                             field: ScalarField,
                             phi_hat: ScalarField) -> ConsistencyScore:
        """
        Convenience method: compute consistency using tracker state.
        """
        omega = tracker.get_averaged()
        sigma = tracker.shell_extractor.extract(field)
        rho_q = tracker.shell_extractor.curvature_field(field)
        
        return self.compute(omega, sigma, rho_q, phi_hat, field)


def compute_consistency_matrix(omega: ScalarField,
                                sigma: np.ndarray,
                                rho_q: ScalarField,
                                candidates: Dict[str, ScalarField],
                                weights: Optional[Dict] = None) -> Dict[str, ConsistencyScore]:
    """
    Compute consistency scores for multiple candidates.
    
    Returns:
        Dictionary mapping candidate name to consistency score.
    """
    weights = weights or {}
    cc = CollapseConsistency(
        alpha=weights.get('alpha', 1.0),
        beta=weights.get('beta', 1.0),
        gamma=weights.get('gamma', 0.5),
        delta=weights.get('delta', 0.0)
    )
    
    scores = {}
    for name, phi_hat in candidates.items():
        scores[name] = cc.compute(omega, sigma, rho_q, phi_hat)
    
    return scores


def best_match(scores: Dict[str, ConsistencyScore]) -> Tuple[str, ConsistencyScore]:
    """
    Find the candidate with lowest consistency score (best match).
    """
    best_name = min(scores, key=lambda k: scores[k].total)
    return best_name, scores[best_name]


class AdaptiveConsistency(CollapseConsistency):
    """
    Collapse consistency with adaptive weights based on observed dynamics.
    
    Automatically adjusts weights based on what features are present.
    """
    
    def __init__(self, base_alpha: float = 1.0, base_beta: float = 1.0,
                 base_gamma: float = 0.5, base_delta: float = 0.3):
        super().__init__(base_alpha, base_beta, base_gamma, base_delta)
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.base_gamma = base_gamma
        self.base_delta = base_delta
    
    def adapt_weights(self, omega: ScalarField, sigma: np.ndarray,
                      rho_q: ScalarField) -> Dict[str, float]:
        """
        Adjust weights based on observed dynamics.
        
        - If lots of shells: increase beta
        - If high memory variance: increase alpha
        - If strong curvature: increase gamma
        """
        # Shell strength
        shell_coverage = np.mean(sigma)
        beta_factor = 1.0 + shell_coverage * 2.0
        
        # Memory variance
        omega_std = np.std(omega.data)
        alpha_factor = 1.0 + omega_std * 2.0
        
        # Curvature strength
        curv_strength = np.std(rho_q.data)
        gamma_factor = 1.0 + curv_strength
        
        return {
            'alpha': self.base_alpha * alpha_factor,
            'beta': self.base_beta * beta_factor,
            'gamma': self.base_gamma * gamma_factor,
            'delta': self.base_delta
        }
    
    def compute_adaptive(self, omega: ScalarField, sigma: np.ndarray,
                         rho_q: ScalarField, phi_hat: ScalarField,
                         observed_field: Optional[ScalarField] = None) -> ConsistencyScore:
        """Compute with adaptive weights."""
        weights = self.adapt_weights(omega, sigma, rho_q)
        
        self.alpha = weights['alpha']
        self.beta = weights['beta']
        self.gamma = weights['gamma']
        self.delta = weights['delta']
        
        return self.compute(omega, sigma, rho_q, phi_hat, observed_field)


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALIZED CONSISTENCY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def transformation_consistency(field_before: ScalarField,
                                field_after: ScalarField,
                                transform_field: ScalarField) -> float:
    """
    Check if applying transform to before gives after.
    
    C_transform = ||T(Φ_before) - Φ_after||²
    """
    transformed = field_before + transform_field
    return float(np.mean((transformed.data - field_after.data)**2))


def spectral_consistency(field: ScalarField, target: ScalarField) -> float:
    """
    Compare fields in frequency space.
    
    C_spectral = ||FFT(Φ) - FFT(Φ*)||²
    """
    fft_field = np.fft.fft2(field.data)
    fft_target = np.fft.fft2(target.data)
    
    # Compare magnitudes
    mag_field = np.abs(fft_field)
    mag_target = np.abs(fft_target)
    
    # Normalize
    mag_field = mag_field / (np.max(mag_field) + 1e-8)
    mag_target = mag_target / (np.max(mag_target) + 1e-8)
    
    return float(np.mean((mag_field - mag_target)**2))


def symmetry_consistency(field: ScalarField, symmetry_type: str) -> float:
    """
    Check how well field satisfies a symmetry.
    
    Types:
        'h_flip': Horizontal flip
        'v_flip': Vertical flip
        'rot90': 90° rotation
        'rot180': 180° rotation
    """
    data = field.data
    
    if symmetry_type == 'h_flip':
        transformed = np.fliplr(data)
    elif symmetry_type == 'v_flip':
        transformed = np.flipud(data)
    elif symmetry_type == 'rot90':
        transformed = np.rot90(data)
    elif symmetry_type == 'rot180':
        transformed = np.rot90(data, 2)
    else:
        raise ValueError(f"Unknown symmetry: {symmetry_type}")
    
    return float(np.mean((data - transformed)**2))
