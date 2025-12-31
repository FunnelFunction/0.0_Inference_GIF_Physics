# Build Specification: Inference GIF Physics
## ARC-AGI Solver via Geometric Field Inference

---

## 1. Mission Statement

**Goal**: Solve ARC-AGI puzzles without training, using scalar field dynamics to INFER the transformation rule from examples.

**Core Insight**: The examples aren't data to learn from—they're "vibrational hints" that initialize a field whose natural attractor IS the transformation.

---

## 2. Theoretical Foundation

### 2.1 The Paradigm Shift

| Traditional ML | This System |
|----------------|-------------|
| Learn rule θ from examples | Infer Φ* from field structure |
| Optimize P(output\|input, θ) | Evolve to natural attractor |
| Needs many examples | Uses geometric encoding |
| Discrete symbolic rules | Continuous field dynamics |

### 2.2 Core Equations

**Resolution (Previous Work):**
```
Given Φ*, evolve Φ → Φ* via:
∂Φ/∂t = F[Φ] - lr(Φ - Φ*)
```

**Inference (This Work):**
```
Infer Φ* from examples, then resolve:

Φ* = argmin_{Φ̂ ∈ Λ} C(Ω^, Σ, ρq | Φ̂)

where:
  Λ = attractor library (transformation templates)
  C = collapse consistency function
  Ω^ = gradient memory
  Σ = shell structures
  ρq = curvature tension
```

### 2.3 The ARC Pipeline

```
ARC Examples: {(A₁,B₁), (A₂,B₂), (A₃,B₃)}
                    ↓
            Grid → Field Lifting
                    ↓
            Transformation Encoding
            ΔΦᵢ = Lift(Bᵢ) - Lift(Aᵢ)
                    ↓
            Compose Test Input
            Φ₀ = Compose(A_test, {ΔΦᵢ})
                    ↓
            Pure CSS Evolution (no target)
            ∂Φ/∂t = η∇²Φ + μΦ(1-Φ²)
                    ↓
            Extract: {Ω^, Σ, ρq}
                    ↓
            Match against Λ → Infer Φ*
                    ↓
            Resolve: Φ → Φ*
                    ↓
            Decode: B_test = Field → Grid
```

---

## 3. Mathematical Components

### 3.1 Grid ↔ Field Encoding

**Lift: Grid → Field**
```python
def lift(grid: np.ndarray) -> ScalarField:
    """
    Convert ARC grid (integers 0-9) to scalar field.
    
    Encoding options:
    1. Direct: Φ(i,j) = grid[i,j] / 9  (normalized to [0,1])
    2. Signed: Φ(i,j) = (grid[i,j] - 4.5) / 4.5  (centered at 0)
    3. One-hot expansion: 10 channels, one per color
    4. Spectral: FFT of grid, phase + magnitude
    """
    # Use signed encoding for bistable compatibility
    return (grid.astype(float) - 4.5) / 4.5
```

**Decode: Field → Grid**
```python
def decode(field: ScalarField) -> np.ndarray:
    """
    Convert scalar field back to ARC grid.
    
    1. Scale: Φ ∈ [-1,1] → [0,9]
    2. Round to nearest integer
    3. Clamp to valid range
    """
    scaled = (field * 4.5 + 4.5)
    return np.clip(np.round(scaled), 0, 9).astype(int)
```

### 3.2 Transformation Signature

**Compute transformation signature from example pair:**
```python
def transformation_signature(A: Grid, B: Grid) -> TransformField:
    """
    ΔΦ = Lift(B) - Lift(A)
    
    This encodes "what changed" as a field difference.
    """
    Φ_A = lift(A)
    Φ_B = lift(B)
    return Φ_B - Φ_A
```

**Compose signatures from multiple examples:**
```python
def compose_signatures(deltas: List[TransformField]) -> TransformField:
    """
    Combine multiple ΔΦᵢ into single transformation template.
    
    Options:
    1. Average: ΔΦ = mean(ΔΦᵢ)
    2. Spectral consensus: Keep modes present in all
    3. Weighted by consistency
    """
    # Start with average
    return np.mean(deltas, axis=0)
```

### 3.3 Field Evolution (Pure Physics)

**Modified Swift-Hohenberg without target:**
```python
def evolve_pure(Φ: ScalarField, dt: float, params: dict) -> dict:
    """
    ∂Φ/∂t = rΦ - (k₀² + ∇²)²Φ - gΦ³
    
    No learning term. Let the field find its natural attractor.
    """
    r = params.get('r', 0.3)
    g = params.get('g', 1.0)
    k0 = params.get('k0', 1.0)
    
    # Compute operators
    lap = laplacian(Φ)
    bilap = laplacian(lap)
    
    # Generalized Swift-Hohenberg
    k0_sq = k0 ** 2
    linear_op = (k0_sq ** 2) * Φ + 2 * k0_sq * lap + bilap
    reaction = Φ * (r - g * Φ ** 2)
    
    # Evolution
    dΦdt = reaction - linear_op
    
    return Φ + dt * dΦdt
```

### 3.4 Shell Extraction (Σ)

```python
def extract_shells(Φ: ScalarField, threshold: float = 1.0) -> ShellMask:
    """
    Σ = {(i,j) : |∇Φ|(i,j) > τ}
    
    Shells are high-gradient boundaries between domains.
    """
    grad_mag = gradient_magnitude(Φ)
    
    # Adaptive threshold
    mean = np.mean(grad_mag)
    std = np.std(grad_mag)
    τ = mean + threshold * std
    
    return grad_mag > τ
```

**Shell Features:**
```python
def shell_features(Σ: ShellMask, Φ: ScalarField) -> dict:
    """
    Extract geometric features from shells:
    - Count: number of shell pixels
    - Topology: connected components
    - Curvature: ∇²Φ at shell locations
    - Orientation: gradient direction histogram
    """
    return {
        'count': np.sum(Σ),
        'components': count_connected_components(Σ),
        'curvature_mean': np.mean(laplacian(Φ)[Σ]),
        'curvature_std': np.std(laplacian(Φ)[Σ]),
        'orientation_hist': gradient_orientation_histogram(Φ, Σ)
    }
```

### 3.5 Memory Tracking (Ω^)

```python
def update_memory(Ω: ScalarField, Φ: ScalarField, Σ: ShellMask, 
                  γ: float = 0.95) -> ScalarField:
    """
    Ω^_{n+1} = γ·Ω^_n + (1-γ)·ρq·|Φ|
    
    Memory accumulates where shells form and field is active.
    """
    ρq = Σ.astype(float)
    return γ * Ω + (1 - γ) * ρq * np.abs(Φ)
```

### 3.6 Collapse Consistency Function (C)

```python
def collapse_consistency(Ω: ScalarField, Σ: ShellMask, ρq: ScalarField,
                         Φ_hat: ScalarField, weights: dict) -> float:
    """
    C(Ω^, Σ, ρq | Φ̂) = α||∇Φ̂ - Ω^_avg||² + β·MatchShells(Σ, Φ̂) + γ||∇²Φ̂ - ρq||
    
    Scores how well candidate Φ̂ explains observed dynamics.
    Lower = better match.
    """
    α = weights.get('alpha', 1.0)
    β = weights.get('beta', 1.0)
    γ = weights.get('gamma', 1.0)
    
    # Term 1: Gradient consistency
    grad_Φ_hat = gradient_magnitude(Φ_hat)
    Ω_avg = Ω / (np.max(Ω) + 1e-8)  # Normalize
    grad_term = α * np.mean((grad_Φ_hat - Ω_avg) ** 2)
    
    # Term 2: Shell matching
    Σ_hat = extract_shells(Φ_hat)
    shell_term = β * shell_distance(Σ, Σ_hat)
    
    # Term 3: Curvature consistency
    lap_Φ_hat = laplacian(Φ_hat)
    curv_term = γ * np.mean((lap_Φ_hat - ρq) ** 2)
    
    return grad_term + shell_term + curv_term
```

### 3.7 Shell Matching Metric

```python
def shell_distance(Σ1: ShellMask, Σ2: ShellMask) -> float:
    """
    Compare two shell configurations.
    
    Metrics:
    1. IoU: Intersection over Union
    2. Hausdorff: Maximum closest-point distance
    3. Feature distance: Compare shell_features
    """
    # Intersection over Union
    intersection = np.sum(Σ1 & Σ2)
    union = np.sum(Σ1 | Σ2)
    iou = intersection / (union + 1e-8)
    
    return 1.0 - iou  # Distance = 1 - similarity
```

### 3.8 Attractor Library (Λ)

```python
class AttractorLibrary:
    """
    Library of known transformation templates.
    
    Each attractor is a field pattern that represents a transformation type.
    """
    
    def __init__(self):
        self.attractors = {}
        self._build_base_attractors()
    
    def _build_base_attractors(self):
        """
        Initialize with ARC-common transformations:
        - rotate_90, rotate_180, rotate_270
        - flip_h, flip_v
        - identity
        - scale_2x, scale_0.5x
        - color_swap patterns
        """
        self.attractors['identity'] = self._identity_attractor()
        self.attractors['rotate_90'] = self._rotation_attractor(90)
        self.attractors['rotate_180'] = self._rotation_attractor(180)
        self.attractors['flip_h'] = self._flip_attractor('h')
        self.attractors['flip_v'] = self._flip_attractor('v')
        # ... more
    
    def match(self, Ω: ScalarField, Σ: ShellMask, ρq: ScalarField) -> str:
        """
        Find best matching attractor.
        
        Returns attractor ID with lowest collapse consistency score.
        """
        scores = {}
        for name, Φ_hat in self.attractors.items():
            scores[name] = collapse_consistency(Ω, Σ, ρq, Φ_hat, {})
        
        return min(scores, key=scores.get)
```

---

## 4. ARC-AGI Integration

### 4.1 Data Format

ARC puzzles are JSON:
```json
{
  "train": [
    {"input": [[0,0,1],[0,1,0],[1,0,0]], "output": [[1,0,0],[0,1,0],[0,0,1]]}
  ],
  "test": [
    {"input": [[0,1,0],[1,0,1],[0,1,0]], "output": [[0,1,0],[1,0,1],[0,1,0]]}
  ]
}
```

### 4.2 Solver Pipeline

```python
def solve_arc_puzzle(puzzle: dict) -> List[Grid]:
    """
    Main solver entry point.
    """
    # 1. Extract examples
    train = puzzle['train']
    test = puzzle['test']
    
    # 2. Compute transformation signatures from training examples
    signatures = []
    for example in train:
        A = np.array(example['input'])
        B = np.array(example['output'])
        Δ = transformation_signature(A, B)
        signatures.append(Δ)
    
    # 3. Compose into unified transformation field
    T = compose_signatures(signatures)
    
    # 4. For each test input
    results = []
    for test_case in test:
        A_test = np.array(test_case['input'])
        
        # 5. Initialize field with transformation signature
        Φ_A = lift(A_test)
        Φ_0 = Φ_A + T  # Apply transformation hint
        
        # 6. Evolve under pure physics
        Φ, Ω, Σ = evolve_to_attractor(Φ_0, max_steps=100)
        
        # 7. Optionally: match against library and refine
        # attractor_id = library.match(Ω, Σ, ρq)
        # Φ = apply_attractor(Φ, attractor_id)
        
        # 8. Decode to grid
        B_test = decode(Φ)
        results.append(B_test)
    
    return results
```

### 4.3 Evolution to Attractor

```python
def evolve_to_attractor(Φ_0: ScalarField, max_steps: int = 100,
                         dt: float = 0.1, tolerance: float = 1e-4) -> tuple:
    """
    Evolve field until it reaches a stable attractor.
    """
    Φ = Φ_0.copy()
    Ω = np.zeros_like(Φ)
    
    params = {'r': 0.3, 'g': 1.0, 'k0': 1.0}
    
    for step in range(max_steps):
        Φ_prev = Φ.copy()
        
        # Evolve
        Φ = evolve_pure(Φ, dt, params)
        
        # Extract shells
        Σ = extract_shells(Φ)
        
        # Update memory
        Ω = update_memory(Ω, Φ, Σ)
        
        # Check convergence
        change = np.max(np.abs(Φ - Φ_prev))
        if change < tolerance:
            break
    
    ρq = laplacian(Φ)
    return Φ, Ω, Σ
```

---

## 5. Implementation Plan

### Phase 1: Core Infrastructure
- [ ] `core/scalar_field.py` - Field class with operators
- [ ] `core/field_dynamics.py` - CSS/SH evolution
- [ ] `core/shell_extractor.py` - Σ extraction
- [ ] `core/omega_tracker.py` - Memory system
- [ ] `core/grid_codec.py` - Grid ↔ Field

### Phase 2: Inference Engine
- [ ] `core/collapse_consistency.py` - C function
- [ ] `core/match_shells.py` - Shell comparison metrics
- [ ] `core/attractor_library.py` - Λ with base transformations
- [ ] `core/inference_solver.py` - Main solver

### Phase 3: ARC Integration
- [ ] `experiments/arc_loader.py` - Load ARC JSON
- [ ] `experiments/arc_solver.py` - Full pipeline
- [ ] `experiments/evaluate.py` - Score against ground truth

### Phase 4: Optimization
- [ ] Parallel evolution for speed
- [ ] Adaptive k₀ from grid size
- [ ] Multi-scale approach (coarse → fine)

---

## 6. Key Challenges

### 6.1 The Λ Bootstrap Problem
**Issue**: Where do attractors come from?
**Solution**: 
1. Start with hand-coded basic transformations
2. Discover new attractors from solved puzzles
3. Build library incrementally

### 6.2 Grid Size Variation
**Issue**: ARC grids vary in size (1×1 to 30×30)
**Solution**:
1. Pad to fixed size with neutral value
2. Or: Scale k₀ inversely with grid size
3. Or: Use relative coordinates

### 6.3 Color Encoding
**Issue**: ARC uses 10 colors (0-9) with semantic meaning
**Solution**:
1. Multi-channel fields (one per color)
2. Or: Treat as continuous variable
3. Or: Binary fields per color threshold

### 6.4 Transformation Composition
**Issue**: ARC rules can be compound (rotate AND recolor)
**Solution**:
1. Decompose into sequential transformations
2. Or: Learn compound attractors
3. Or: Use transformation algebra

---

## 7. Success Criteria

| Metric | Target |
|--------|--------|
| Training examples used | 2-3 per puzzle (given) |
| External training | ZERO |
| Accuracy on evaluation set | >50% (beats GPT-4) |
| Time per puzzle | <60 seconds |

---

## 8. Appendix: ARC-AGI Data

**Source**: https://github.com/fchollet/ARC-AGI/releases/tag/v1.0.2

**Format**:
- 400 training puzzles
- 400 evaluation puzzles
- Each puzzle: 2-10 train examples + 1-3 test cases
- Grids: 1×1 to 30×30
- Colors: 0-9 (10 values)

---

*Document Version: 1.0*
*Last Updated: December 31, 2024*
