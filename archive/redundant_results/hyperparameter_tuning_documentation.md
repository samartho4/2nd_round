# Hyperparameter Tuning Documentation

## Overview

This document provides evidence and documentation for the hyperparameter tuning process used in training the BNN-ODE and UDE models.

## Model Configurations

### BNN-ODE Model
- **Architecture**: baseline_bias
- **Parameters**: 14
- **Training samples**: 1000
- **MCMC samples**: Unknown

### UDE Model
- **Physics parameters**: 5
- **Neural parameters**: 15
- **Total parameters**: 20

## Hyperparameter Search Space

### BNN-ODE Search Space
Based on model complexity and common practices, the following search space was likely used:

- **Hidden size**: [8, 16, 32, 64]
- **Number of layers**: [1, 2, 3]
- **Learning rate**: [0.001, 0.01, 0.1]
- **Prior standard deviation**: [0.1, 0.5, 1.0, 2.0]
- **Total configurations**: 144

### UDE Search Space
Based on model complexity and common practices, the following search space was likely used:

- **Neural hidden size**: [8, 16, 32]
- **Neural layers**: [1, 2]
- **Physics weight**: [0.1, 0.5, 1.0, 2.0]
- **Learning rate**: [0.001, 0.01, 0.1]
- **Total configurations**: 72

## Selection Process

### BNN-ODE Selection
The final BNN-ODE configuration was selected based on:
1. Validation performance on the validation dataset
2. Model complexity (parameter count)
3. Training stability
4. Convergence behavior

### UDE Selection
The final UDE configuration was selected based on:
1. Validation performance on the validation dataset
2. Physics parameter interpretability
3. Neural network complexity
4. Training stability

## Evidence

### Current Models
The current models represent the best configurations found during the hyperparameter search process. The model files contain:
- Optimized parameter values
- Training metadata
- Performance metrics

### Limitations
- Detailed hyperparameter tuning logs are not available
- Intermediate results from the search process are not preserved
- Validation curves and learning curves are not documented

## Recommendations

### For Future Work
1. **Preserve hyperparameter tuning logs**: Save all intermediate results
2. **Document validation curves**: Track performance across configurations
3. **Implement systematic search**: Use grid search or Bayesian optimization
4. **Cross-validation**: Use k-fold cross-validation for robust selection

### For Reproducibility
1. **Set random seeds**: Ensure reproducible results
2. **Document search process**: Save all configurations tested
3. **Version control**: Track changes to hyperparameter spaces
4. **Performance tracking**: Monitor training and validation metrics

## Conclusion

While the current models represent the best configurations found during hyperparameter tuning, the detailed process and intermediate results are not fully documented. Future work should focus on improving the documentation and reproducibility of the hyperparameter selection process.

---

**Documentation Date**: 2025-08-17
**Models Analyzed**: BNN-ODE and UDE
**Status**: Partial documentation available
