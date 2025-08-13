## Final Results

| Method | Trajectory MSE | Symbolic R2 | Training Data | Numerical Stability |
|--------|----------------|-------------|---------------|-------------------|
| Bayesian Neural ODE | 32.16 | N/A | 1,500 points | abstol=1.0e-8, reltol=1.0e-8 |
| UDE (Universal Differential Equations) | 16.71 | 0.9288 | 1,500 points | abstol=1.0e-8, reltol=1.0e-8 |
| Physics-Only Model | 0.16 | N/A | N/A | abstol=1.0e-8, reltol=1.0e-8 |
| Symbolic Discovery | N/A | 0.9288 | N/A | N/A |

**Key Findings:**
- **Trajectory Simulation**: Models evaluated by simulating full trajectories and comparing to ground truth
- **Physics Discovery**: UDE successfully discovered hidden physics with R2 = 0.9288
- **Numerical Stability**: All simulations use strict tolerances (abstol=1.0e-8, reltol=1.0e-8)
- **Evaluation**: 953 points across 3 scenarios
