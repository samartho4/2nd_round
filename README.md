# Microgrid Bayesian Neural ODE Control

This repository demonstrates physics discovery with Universal Differential Equations (UDEs) and Bayesian Neural ODEs, now upgraded for NeurIPS-level rigor.

## Canonical pipeline (current)

- One-command: `./bin/reproduce.sh`
- Key scripts: `scripts/train.jl`, `scripts/evaluate.jl`, `scripts/generate_results_summary.jl`, `scripts/generate_symbolic_table.jl`, `scripts/generate_figures.jl`
- Rigor add-ons: `scripts/statistical_validation.jl`, `comprehensive_baselines.jl`, `dataset_analysis.jl`, `ablation_comprehensive.jl`, `generalization_study.jl`, `physics_validation.jl`, `realistic_validation.jl`, `computational_benchmarks.jl`

## Redundant (archived for reference)

The following legacy/duplicate scripts were moved to `scripts/redundant/` or `src/redundant/` to keep the active pipeline clean:

- `scripts/redundant/generate_figures_complex.jl`
- `scripts/redundant/neurips_statistical_evaluation.jl`
- `scripts/redundant/fresh_evaluation.jl`
- `scripts/redundant/evaluate_complex.jl`
- `scripts/redundant/cross_validation_training.jl`
- `scripts/redundant/ultra_stable_train.jl`
- `scripts/redundant/ultra_stable_evaluation.jl`
- `scripts/redundant/train_complex.jl`
- `src/redundant/baselines.jl`

Use the canonical pipeline above for all new experiments and figures.



