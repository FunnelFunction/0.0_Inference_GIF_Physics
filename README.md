# Inference GIF Physics

## Solving ARC-AGI via Geometric Field Inference

> **"When a field knows where to go, it does not need to be told."**

This repository extends [Executable GIF Physics](https://github.com/FunnelFunction/0.0_Executable_GIF_Physics) from **resolution** to **inference**:

| Previous Work | This Work |
|---------------|-----------|
| Given Φ*, resolve Φ → Φ* | Infer Φ* from examples, then resolve |
| Target known | Target discovered |
| Resolution | Inference + Resolution |

---

## The Core Insight

Traditional ML asks: "What rule θ explains these examples?"

We ask: "What attractor does this field evolve toward?"

**The examples aren't training data—they're vibrational hints.**

---

## Mathematical Foundation

### 1. Collapse Consistency Equation

```
Φ* = argmin_{Φ̂ ∈ Λ} C(Ω^, Σ, ρq | Φ̂)
```

Where:
- `Λ` = Library of transformation attractors
- `Ω^` = Accumulated gradient memory
- `Σ` = Shell structures (boundaries)
- `ρq` = Curvature field
- `C` = Collapse consistency function

### 2. The Consistency Function

```
C(Ω^, Σ, ρq | Φ̂) = α||∇Φ̂ - Ω^_avg||² + β·d(Σ, Σ̂) + γ||∇²Φ̂ - ρq||²
```

This asks: **"If Φ̂ were the target, would it produce the dynamics we observed?"**

### 3. Field Initialization from Examples

Given examples `{(A₁,B₁), (A₂,B₂), (A₃,B₃)}`:

```
ΔΦᵢ = Lift(Bᵢ) - Lift(Aᵢ)  // Transformation signature
Φ₀ = Compose(A_test, {ΔΦᵢ})  // Initialize test with signature
```

### 4. Pure Evolution (No Target)

```
∂Φ/∂t = η∇²Φ + μΦ(1-Φ²) - λ∇⁴Φ
```

Let the field find its natural attractor:
```
Φ* = lim_{t→∞} Φ(t)
```

---

## Architecture

```
0.0_Inference_GIF_Physics/
├── core/
│   ├── scalar_field.py      # Field data structure
│   ├── field_dynamics.py    # PDE evolution operators
│   ├── shell_extractor.py   # Σ extraction
│   ├── omega_tracker.py     # Ω^ memory
│   ├── collapse_consistency.py  # C function
│   ├── attractor_library.py # Λ transformations
│   ├── grid_codec.py        # ARC ↔ Field conversion
│   └── inference_solver.py  # Main solver
├── experiments/
│   └── arc_loader.py        # Load ARC puzzles
├── codex/
│   ├── Book_I_Resolution.md
│   ├── Book_II_Inference.md
│   └── Book_III_Becoming.md
└── BUILD_SPECIFICATION.md
```

---

## Usage

### Quick Start

```python
from core import InferenceSolver, SolverConfig
from experiments.arc_loader import ARCLoader

# Load solver
solver = InferenceSolver()

# Load ARC puzzle
loader = ARCLoader()
puzzle = loader.load_puzzle('00d62c1b', split='training')

# Solve
predictions = solver.solve(puzzle)

# Check accuracy
for i, pred in enumerate(predictions):
    target = puzzle['test'][i].get('output')
    if target:
        print(f"Test {i}: {np.array_equal(pred, target)}")
```

### Custom Configuration

```python
config = SolverConfig(
    encoding_method='signed',
    max_evolution_steps=100,
    evolution_dt=0.05,
    use_attractor_library=True,
    dynamics_params={
        'eta': 0.1,
        'mu': 0.5,
        'lambda': 0.02
    }
)

solver = InferenceSolver(config)
```

### Evaluate on Dataset

```python
from core.inference_solver import evaluate_solver

loader = ARCLoader()
puzzles = loader.load_sample(n=50, split='training')

results = evaluate_solver(solver, puzzles, verbose=True)
print(f"Accuracy: {results['accuracy']:.1%}")
```

---

## The ARC-AGI Pipeline

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
            Pure Evolution (no target)
            ∂Φ/∂t = F[Φ]
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

## Key Innovations

### 1. No Training Required
The system doesn't learn from data. It infers structure from geometry.

### 2. Transformation as Field Dynamics
Rules aren't symbolic—they're attractors in field space.

### 3. Collapse Consistency
We don't ask "what rule?" We ask "what target explains these dynamics?"

### 4. Natural Attractors
The field evolves toward stable states without being told what they are.

---

## Attractor Library

Built-in transformations:

| Name | Type | Description |
|------|------|-------------|
| `identity` | Identity | No change |
| `rotate_90` | Rotation | 90° counterclockwise |
| `rotate_180` | Rotation | 180° rotation |
| `rotate_270` | Rotation | 270° counterclockwise |
| `flip_h` | Reflection | Horizontal flip |
| `flip_v` | Reflection | Vertical flip |
| `flip_diag` | Reflection | Diagonal transpose |
| `invert` | Color | Negate values |
| `complement` | Color | Color complement |
| `tile_2x2` | Tiling | Tile into 2x2 |
| `scale_2x` | Scaling | Double resolution |

---

## Requirements

```
numpy
scipy
```

Optional:
```
matplotlib  # Visualization
```

---

## Citation

```bibtex
@software{inference_gif_physics,
  title={Inference GIF Physics: Solving ARC-AGI via Geometric Field Inference},
  author={Knight, Armstrong and Khan, Abdullah},
  year={2024},
  url={https://github.com/FunnelFunction/0.0_Inference_GIF_Physics}
}
```

---

## Related Work

- [Executable GIF Physics](https://github.com/FunnelFunction/0.0_Executable_GIF_Physics) - Resolution via self-resolving dynamical systems
- [ARC-AGI](https://github.com/fchollet/ARC-AGI) - The benchmark dataset
- [Intent Tensor Theory](https://github.com/intent-tensor-theory/0.0_Coding_Principals_Intent_Tensor_Theory) - Theoretical foundation

---

## License

MIT

---

*"The field resolves because it must. Not from knowing—but from collapsing."*
— Book of Φ
