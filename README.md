# Microgrid Bayesian Neural ODE Control

This repository demonstrates physics discovery with Universal Differential Equations (UDEs) and Bayesian Neural ODEs, upgraded for NeurIPS-level rigor.

## Quickstart

- Setup once: `bin/mg setup` (or `make setup`)
- Full reproduce: `bin/mg repro` (or `make repro`)
- Common tasks:
  - Data: `bin/mg data`
  - Train: `bin/mg train`
  - Eval: `bin/mg eval`
  - Stats: `bin/mg stats`
  - Baselines: `bin/mg baselines`
  - Dataset analysis: `bin/mg dataset`
  - Figures: `bin/mg figs`
  - Benchmarks: `bin/mg bench`
  - Results summary: `bin/mg results`
  - Verify: `bin/mg verify`

## Structure (active)

- `scripts/`
  - Core: `train.jl`, `evaluate.jl`, `generate_results_summary.jl`, `generate_symbolic_table.jl`, `generate_figures.jl`
  - Rigor: `statistical_validation.jl`, `comprehensive_baselines.jl`, `dataset_analysis.jl`, `ablation_comprehensive.jl`, `generalization_study.jl`, `physics_validation.jl`, `realistic_validation.jl`, `computational_benchmarks.jl`
  - Orchestration: `run_full_pipeline.jl`
- `src/`: core modules (`microgrid_system.jl`, `neural_ode_architectures.jl`, `statistical_framework.jl`, ...)
- `bin/mg`: simple task runner
- `Makefile`: convenience targets
- `paper/`, `outputs/`: figures and results
- `test/`: tests (`Pkg.test()` via `test/runtests.jl`)

## Archived

Legacy/duplicate scripts live in `scripts/redundant/` and `src/redundant/`. Use the active structure above.



