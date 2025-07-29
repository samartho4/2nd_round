# Project Structure & Technical Details

## Overview

This project implements Bayesian Neural ODEs for microgrid dynamics modeling. The goal is to learn uncertain dynamics while maintaining interpretability.

## Core Components

### 1. Data Generation (`scripts/make_data_multiscenario.jl`)
- Generates 45,000+ data points across 25 scenarios
- Each scenario has different parameters and initial conditions
- Time range: 60-72 hours per scenario
- Features: time, x1, x2 (grid states), scenario ID

### 2. Bayesian Neural ODE (`scripts/train_bayesian_node.jl`)
- **Architecture**: 3 inputs → 2 hidden → 2 outputs (10 parameters)
- **Framework**: Turing.jl for Bayesian inference
- **Sampler**: NUTS with 1000+ iterations
- **Output**: State derivatives [dx1/dt, dx2/dt]

### 3. Universal Differential Equation (`scripts/train_ude_working.jl`)
- **Hybrid approach**: Combines physics with neural networks
- **Physics parameters**: 5 parameters (ηin, ηout, α, β, γ)
- **Neural parameters**: 15 additional parameters
- **Goal**: Learn unknown dynamics while preserving physics

### 4. Symbolic Regression (`scripts/symbolic_regression_simple.jl`)
- **Purpose**: Extract interpretable equations from neural networks
- **Method**: Symbolic regression on neural network outputs
- **Result**: Perfect extraction (R² = 1.0)

### 5. Control Integration (`scripts/control_integration.jl`)
- **PID Control**: Standard proportional-integral-derivative
- **Uncertainty-Aware Control**: Adapts to uncertainty levels
- **Robust Control**: Conservative approach for worst-case scenarios

## File Structure

```
├── scripts/                    # Main analysis scripts
│   ├── make_data_multiscenario.jl    # Data generation
│   ├── train_bayesian_node.jl        # Bayesian NODE
│   ├── train_ude_working.jl          # UDE training
│   ├── symbolic_regression_simple.jl  # Symbolic extraction
│   ├── control_integration.jl         # Control policies
│   └── final_publication_results.jl   # Results summary
├── src/
│   ├── Microgrid.jl                  # Physics model
│   └── NeuralNODEArchitectures.jl    # Neural networks
├── data/
│   ├── train_improved.csv            # Training data (30k+ points)
│   ├── val_improved.csv              # Validation data (7k+ points)
│   ├── test_improved.csv             # Test data (7k+ points)
│   └── scenarios/                    # Individual scenario data
├── paper/
│   ├── figures/                      # Generated plots
│   └── results/                      # Performance metrics
└── checkpoints/                      # Saved model results
```

## Key Results

### Performance Metrics
- **Physics Model**: R² = 99.99%, MSE = 0.0011 (excellent baseline)
- **Bayesian NODE**: R² = 56.5%, MSE = 15.36, Coverage = 32.2%
- **UDE**: R² ≈ 50%, MSE ≈ 20, Coverage ≈ 40%
- **Symbolic Discovery**: R² = 100%, MSE = 0.0 (perfect extraction)

### Dataset Statistics
- **Total Points**: 45,283
- **Scenarios**: 25 (5 parameter sets × 5 initial conditions)
- **Time Range**: 60-72 hours per scenario
- **Features**: time, x1, x2, scenario

## Technical Challenges

### 1. Neural Network Training
- **Issue**: Limited to small networks (10 parameters)
- **Solution**: Manual implementation instead of Lux.jl
- **Result**: Working but underperforming compared to physics

### 2. Uncertainty Quantification
- **Issue**: Coverage (32.2%) below ideal (95%)
- **Cause**: Underconfidence in uncertainty estimates
- **Impact**: Less reliable for safety-critical applications

### 3. MCMC Sampling
- **Issue**: Complex parameter extraction from chains
- **Solution**: Manual indexing and error handling
- **Result**: Working but computationally expensive

## Implementation Details

### Neural Network Architecture
```julia
function nn_dynamics!(dx, x, p, t)
    # Input: [x1, x2, t]
    inp = [x[1], x[2], t]
    
    # Hidden layer with tanh activation
    h1 = tanh(p[1]*inp[1] + p[2]*inp[2] + p[3]*inp[3])
    h2 = tanh(p[4]*inp[1] + p[5]*inp[2] + p[6]*inp[3])
    
    # Output layer (state derivatives)
    dx[1] = p[7]*h1 + p[8]*h2  # dx1/dt
    dx[2] = p[9]*h1 + p[10]*h2  # dx2/dt
end
```

### Bayesian Model
```julia
@model function bayesian_nn_ode(t, Y, u0)
    # Observation noise prior
    σ ~ truncated(Normal(0.1, 0.05), 0.01, 1.0)
    
    # Neural network parameters prior
    θ ~ MvNormal(zeros(10), 0.1)
    
    # ODE solution and likelihood
    # ... implementation details
end
```

## Limitations & Future Work

### Current Limitations
1. **Small Networks**: Limited to 10-parameter networks
2. **Poor Coverage**: 32.2% vs around 95% 
3. **Underperformance**: Neural methods worse than physics
4. **Limited Scope**: Specific microgrid dynamics only

### Future Improvements
1. **Larger Networks**: Scale to more parameters
2. **Better Priors**: More informative Bayesian priors
3. **Calibration**: Improve uncertainty calibration
4. **Real Data**: Test on actual microgrid measurements

## Reproducibility

- **Random Seeds**: Fixed throughout for consistency
- **Package Versions**: Exact versions in Manifest.toml
- **Saved Results**: All MCMC chains saved in checkpoints/
- **Deterministic**: Same results on different machines

## Dependencies

- **Julia**: 1.11+
- **DifferentialEquations.jl**: ODE solving
- **Turing.jl**: Bayesian inference
- **Plots.jl**: Visualization
- **CSV.jl**: Data loading
- **DataFrames.jl**: Data manipulation

---
