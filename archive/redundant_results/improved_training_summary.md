# Improved Model Training Summary

## Configuration Changes
- Train samples: 2000 (increased from 1000)
- Train warmup: 500 (increased from 200)
- Tuning samples: 500 (increased from 250)
- NUTS target: 0.8 (single target for consistency)
- Solver tolerances: 1e-6 (relaxed for stability)

## Goals
1. Achieve proper Bayesian uncertainty (non-zero parameter std)
2. Improve posterior exploration
3. Better numerical stability
4. More realistic uncertainty quantification

## Status
Training completed with improved settings.
Check parameter uncertainty analysis above for results.
