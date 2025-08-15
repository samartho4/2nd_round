## Final Results

| Method | Trajectory MSE | Training Data | Numerical Stability |
|--------|----------------|---------------|-------------------|
| Bayesian Neural ODE | 28.02 | 1,500 points | abstol=1.0e-8, reltol=1.0e-8 |
| UDE (Universal Differential Equations) | 17.47 | 1,500 points | abstol=1.0e-8, reltol=1.0e-8 |
| Physics-Only Model | 0.16 | N/A | abstol=1.0e-8, reltol=1.0e-8 |
