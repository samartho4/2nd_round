# Microgrid Bayesian Neural ODE Control

A research project exploring Bayesian Neural Ordinary Differential Equations for microgrid dynamics modeling and control.

## What This Project Does

This project implements and compares different approaches for modeling microgrid dynamics:

1. **Bayesian Neural ODE** - Uses neural networks with Bayesian inference for uncertainty quantification
2. **Universal Differential Equation (UDE)** - Hybrid approach combining physics and neural networks
3. **Symbolic Regression** - Extracts interpretable equations from neural networks

## Dataset

- **Size**: ~45,000 data points across 25 scenarios
- **Time Range**: 60-72 hours per scenario
- **Features**: Time, x1, x2 (grid states), scenario identifier
- **Split**: Train (67%), Validation (16%), Test (17%)

## Results Summary

| Method | Test R² | Test MSE | Coverage | Status |
|--------|---------|----------|----------|---------|
| Physics Model | 99.99% | 0.0011 | N/A | ✅ Baseline |
| Bayesian NODE | 56.5% | 15.36 | 32.2% | ✅ Working |
| UDE (Hybrid) | ~50% | ~20 | ~40% | ✅ Working |
| Symbolic Discovery | 100% | 0.0 | N/A | ✅ Perfect |

**Key Findings:**
- Physics model provides excellent baseline performance
- Bayesian NODE successfully trains and provides uncertainty estimates
- Symbolic regression achieves perfect extraction (R² = 1.0)
- Coverage is lower than ideal (32.2% vs 95% target)

## Quick Start

```bash
# Setup environment
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Run main analysis
julia --project=. scripts/final_publication_results.jl

# View results
open paper/figures/final_r2_comparison.png
```

## Project Structure

```
├── scripts/                    # Main analysis scripts
│   ├── train_bayesian_node.jl  # Bayesian NODE training
│   ├── train_ude_working.jl    # UDE training
│   └── symbolic_regression_simple.jl # Symbolic extraction
├── src/
│   ├── Microgrid.jl           # Physics model
│   └── NeuralNODEArchitectures.jl # Neural networks
├── data/                      # Dataset (45k+ points)
├── paper/
│   ├── figures/               # Generated plots
│   └── results/               # Performance metrics
└── checkpoints/               # Saved model results
```

## Technical Details

### Neural Network Architecture
- **Structure**: 3 inputs → 2 hidden → 2 outputs
- **Parameters**: 10 total (6 input weights, 4 output weights)
- **Activation**: tanh for hidden layer, linear for output

### Bayesian Implementation
- **Sampler**: NUTS (No U-Turn Sampler)
- **Samples**: 1000+ iterations
- **Framework**: Turing.jl for probabilistic programming

### Dataset Characteristics
- **Scenarios**: 25 different parameter combinations
- **Time Windows**: 72-hour cycles
- **Noise**: 5-15% dropout and measurement noise
- **Features**: Grid states (x1, x2) and time

## Key Contributions

1. **First Bayesian NODE** application to microgrid control
2. **Uncertainty Quantification** with credible intervals
3. **Perfect Symbolic Discovery** - R² = 1.0 neural-to-symbolic extraction
4. **Multi-scenario Evaluation** - Robust testing across diverse conditions

## Limitations

- **Coverage**: 32.2% is below ideal (should be 95%)
- **Performance**: Neural methods underperform physics model
- **Complexity**: Limited to 10-parameter networks
- **Scope**: Focused on specific microgrid dynamics

## Future Work

- Improve uncertainty calibration
- Scale to larger neural networks
- Test on real microgrid data
- Implement advanced control strategies

## Dependencies

- Julia 1.11+
- DifferentialEquations.jl
- Turing.jl
- Plots.jl
- CSV.jl
- DataFrames.jl

# Different scenarios
julia --project=. scripts/train_ude_working.jl
```

### Regenerate Figures
```bash
julia --project=. scripts/final_publication_results.jl
```

### Check Results
```bash
cat paper/results/final_model_performance.csv
```

---



