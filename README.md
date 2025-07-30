# Microgrid Bayesian Neural ODE Control

Research project implementing Bayesian Neural ODEs for microgrid dynamics modeling.

## Overview

This project compares different approaches for modeling microgrid dynamics:
- Bayesian Neural ODE with uncertainty quantification
- Universal Differential Equation (UDE) hybrid approach
- Symbolic regression for interpretability

## Dataset

- 45,000+ data points across 25 scenarios
- 72-hour time windows per scenario
- Features: time, x1, x2 (grid states)
- Train/Val/Test: 67%/16%/17%

## Results

| Method | Test R² | Test MSE | Coverage |
|--------|---------|----------|----------|
| Baseline (Constant) | 0.0% | 28.05 | N/A |
| Linear Model | 86.2% | 2.0 | N/A |
| Physics Model | 56.5% | 0.067 | N/A |
| Neural Network | 65.0% | 0.15 | 45% |

## Quick Start

```bash
# Setup environment
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Train models
julia --project=. scripts/train.jl

# Evaluate and analyze results
julia --project=. scripts/evaluate.jl

# View results
open paper/figures/model_performance_comparison.png
```

## Project Structure

```
├── scripts/
│   ├── train.jl               # Model training pipeline
│   ├── evaluate.jl            # Evaluation and analysis
│   ├── generate_dataset.jl    # Data generation
│   └── archive/               # Archived scripts
├── src/
│   ├── microgrid_system.jl    # Physics model
│   └── neural_ode_architectures.jl # Neural networks
├── data/                      # Dataset
├── paper/figures/             # Generated plots
└── checkpoints/               # Saved results
```

## Technical Details

- Neural networks: 3→2→2 architecture (10 parameters)
- Bayesian inference: NUTS sampler with Turing.jl
- Dataset: 25 scenarios, 72-hour windows, 5-15% noise

## Limitations

- Coverage below ideal (32.2% vs 95% target)
- Neural methods underperform physics model
- Limited to small networks (10 parameters)

---



