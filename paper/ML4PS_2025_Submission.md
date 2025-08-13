# Microgrid Physics Discovery with Bayesian Neural ODEs and Universal Differential Equations

Authors: [Redacted for review]

## Abstract
We study physics discovery and uncertainty-aware forecasting in a toy microgrid system using Bayesian neural ordinary differential equations (BNN-ODE) and Universal Differential Equations (UDE). We present practical improvements to stabilize training and sampling, diagnose dead networks, and gate symbolic extraction. Our end-to-end reproducibility pipeline produces figures and results tables suitable for ML4PS 2025.

## 1. Introduction
Learning dynamics with neural ODEs suffers from instability, dead networks, and brittle symbolic discovery. We target a microgrid-like system with battery state-of-charge and a proxy grid indicator, combining physics-based structure with learned residuals. Contributions:
- Stabilized Bayesian inference: ADVI warm-starts, tuned NUTS, diagnostics with adaptive retry.
- Dead-network detection and gating of symbolic extraction with sanity checks.
- Tighter priors and output clipping in UDE dynamics to prevent trivial solutions.
- Reproducibility: one-command end-to-end pipeline.

## 2. Methods
### 2.1 System
We use `src/microgrid_system.jl`: SOC and grid proxy evolve under exogenous generation/load with a control schedule (charge/idle/discharge). Physics parameters p = (η_in, η_out, α, β, γ).

### 2.2 Models
- Bayesian Neural ODE (BNN-ODE): replaces full dynamics with a small NN (3→2→2). Priors: θ ~ N(0, σ^2 I).
- UDE: physics dx2 = -α x2 + NN(x1,x2,Pgen,Pload,t) + γ x1, with physics parameters and NN jointly inferred.

### 2.3 Inference and stability
- ADVI warm-start where available; fall back gracefully.
- NUTS with target_accept tuned (0.8–0.9); diagnostics (R-hat, ESS); automatic single adaptive retry.
- Tighter priors for NN weights, output clipping of NN residual for solver robustness.

### 2.4 Dead-network detection and symbolic gating
- Dead-net: low variance of NN outputs over a grid and very low posterior std of NN weights.
- Gate symbolic extraction unless outputs are non-trivial; apply coefficient magnitude checks and require reasonable R².
- Symbolic fit is evaluated against NN outputs; physics validation uses coefficients on Pgen/Pload.

## 3. Experiments
Data: training/validation/test CSVs with multiple scenarios (S1-1…S1-5). We evaluate on held-out windows and simulate trajectories.

### 3.1 Training
We train BNN-ODE and UDE via `scripts/train.jl` with strict solver tolerances. Priors and initialization follow Section 2.3.

### 3.2 Evaluation
We compute MSE/MAE/R² on test data and compare models; simulate trajectories for scenario-level MSE.

### 3.3 Symbolic extraction
We extract a polynomial surrogate for the UDE NN and compute R². We separately verify physics coefficients.

## 4. Results
### 4.1 Performance comparison
The physics-only baseline achieves the lowest MSE on the toy system, as expected. BNN-ODE and UDE achieve moderate errors; UDE improves over BNN-ODE in trajectory MSE.

- See Figure 1 (paper/figures/fig1_performance_comparison.png)

### 4.2 Physics discovery
- UDE NN approximates the nonlinear term; the polynomial surrogate achieves high R² with NN outputs, but coefficient checks indicate physics discovery not validated (Pgen/Pload coefficients ≈ 0).

- See Figure 2 (paper/figures/fig2_physics_discovery.png)

### 4.3 Symbolic extraction success visualization
- Visual summary of symbolic surrogate R².

- See Figure 3 (paper/figures/fig3_ude_symbolic_success.png)

### 4.4 Posterior predictive checks
- We report PPC plots and PIT for calibration diagnostics.

- See PPC (paper/figures/ppc_bayesian_ode.png, paper/figures/ppc_ude.png) and PIT (paper/figures/pit_bnn_x1.png)

### 4.5 Final results table
- See paper/results/final_results_table.md for the markdown table produced by the pipeline.

## 5. Discussion
- Dead networks: mitigated via random init, tighter priors, and output clipping; diagnostics show non-zero output and parameter variance. However, learned residuals can still be simple functions that produce a high R² under polynomial fit without matching true physics.
- MCMC: ADVI warm-start occasionally fails to initialize in this environment; sampling still succeeds with tuned NUTS. Diagnostics and adaptive retries guard against poor chains.
- Symbolic discovery: our gating prevents overclaiming. Physics validation fails on this run (coefficients near zero), emphasizing honest reporting.

## 6. Limitations and Future Work
- Realistic microgrid models and richer datasets needed for robust physics discovery.
- Curriculum learning and pretraining could further reduce trivial residuals.
- More diverse scenarios to test OOD behavior; strengthen PPC and calibration metrics.
- Automate per-scenario breakdown and cross-validation; explore surrogate-based warm starts.

## 7. Reproducibility
Run:
```
./bin/reproduce.sh
```
Outputs:
- Figures: paper/figures
- Results: paper/results, outputs/results

## References
[1] Rackauckas et al., Universal Differential Equations. [2] Turing.jl: Probabilistic Programming in Julia. [3] Chen et al., Neural Ordinary Differential Equations. 