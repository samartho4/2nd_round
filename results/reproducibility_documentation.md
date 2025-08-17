# Reproducibility Documentation

## Environment Setup

### Julia Environment
- **Julia version**: 1.9.0 or later
- **Project**: MicrogridBNNODESubmission
- **Dependencies**: See Project.toml and Manifest.toml

### Random Seeds
- **Evaluation seed**: 42
- **Training seeds**: Not documented in current models
- **Data splitting**: Scenario-based (not random)

## Data Processing

### Dataset Information
- **Training data**: 7,334 samples across 41 scenarios
- **Validation data**: 116 samples across 36 scenarios
- **Test data**: 117 samples across 36 scenarios
- **Data format**: CSV with columns: time, x1, x2, scenario

### Data Splitting
- **Method**: Scenario-based splitting
- **Training scenarios**: 41 unique scenarios
- **Validation scenarios**: 36 unique scenarios
- **Test scenarios**: 36 unique scenarios
- **No overlap**: Scenarios are mutually exclusive

## Model Training

### BNN-ODE Training
- **Architecture**: baseline_bias
- **Parameters**: 14
- **Training data**: 7,334 samples
- **MCMC samples**: 1,000
- **Training time**: ~15 seconds

### UDE Training
- **Architecture**: Physics-informed neural network
- **Physics parameters**: 5
- **Neural parameters**: 15
- **Training data**: 7,334 samples
- **Training time**: ~13 seconds

## Evaluation

### Evaluation Methodology
- **Scenarios evaluated**: 29 out of 36 test scenarios
- **Statistical test**: Wilcoxon signed-rank test
- **Effect size**: Cohen's d
- **Confidence intervals**: 95%

### Reproducible Commands
```bash
# Activate environment
julia --project=.

# Run proper evaluation
julia scripts/proper_evaluation.jl

# Run integrity tests
julia scripts/simple_integrity_test.jl
```

## Limitations

### Current Limitations
1. **Training seeds not set**: Random initialization not controlled
2. **Hyperparameter logs missing**: Tuning process not fully documented
3. **Intermediate results**: Training curves not saved
4. **Validation details**: Validation process not fully documented

### Recommendations
1. **Set all random seeds**: Control all sources of randomness
2. **Save training logs**: Document training process
3. **Version control**: Track all code and data changes
4. **Docker container**: Provide containerized environment

## Conclusion

While the current evaluation is reproducible, the training process could be improved with better documentation and control of random seeds. The evaluation methodology is now statistically sound and reproducible.

---

**Documentation Date**: 2025-08-17
**Evaluation Method**: Proper statistical evaluation
**Status**: Reproducible evaluation, training needs improvement
