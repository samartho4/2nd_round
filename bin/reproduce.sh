#!/usr/bin/env bash
set -euo pipefail

# Reproducibility script for ML4PS 2025 submission
# Usage: ./bin/reproduce.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# 1) Ensure environment is ready
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); println("Environment ready")'

# 2) Train models (BNN-ODE + UDE + Symbolic extraction gating)
julia --project=. -e 'include("scripts/train.jl")'

# 3) Evaluate and compute metrics
julia --project=. -e 'include("scripts/evaluate.jl")'

# 4) Generate figures
julia --project=. -e 'include("scripts/generate_figures.jl")'

# 5) Generate results summary and tables
julia --project=. -e 'include("scripts/generate_results_summary.jl")'
julia --project=. -e 'include("scripts/generate_symbolic_table.jl")'

# 6) Verify results (consistency, unit-style checks)
julia --project=. -e 'include("scripts/verify_results.jl")'

printf "\nReproducibility run complete.\nFigures -> %s/paper/figures\nResults -> %s/paper/results\n" "$ROOT_DIR" "$ROOT_DIR" 