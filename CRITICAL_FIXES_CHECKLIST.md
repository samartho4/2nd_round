# CRITICAL FIXES CHECKLIST - EXECUTE IMMEDIATELY

## ✅ PRIORITY 1 (MUST FIX BEFORE ANY SUBMISSION)

### 1. Remove Unrealistic Results
- [ ] Delete or fix Physics-Only baseline MSE = 0.17
- [ ] Replace with honest baseline comparisons
- [ ] Remove any cherry-picked results

### 2. Fix Data Leakage  
- [ ] Implement temporal splits in create_temporal_splits()
- [ ] Verify no future information in training data
- [ ] Test data should be from later time periods

### 3. Expand Training Data
- [ ] Increase from 1,500 to 15,000+ training points
- [ ] Implement data augmentation in augment_training_data()
- [ ] Document augmentation methodology

### 4. Add Statistical Analysis
- [ ] Implement bootstrap confidence intervals
- [ ] Add multi-seed experiments (minimum 5 seeds)
- [ ] Include statistical significance tests

## ✅ PRIORITY 2 (STRENGTHEN SUBMISSION)

### 5. Proper Baselines
- [ ] Add Linear Regression baseline
- [ ] Add Random Forest baseline  
- [ ] Add LSTM baseline (if applicable)
- [ ] Compare fairly with same data splits

### 6. Physics Discovery Validation
- [ ] Test parameter recovery on known physics
- [ ] Validate extrapolation to new scenarios
- [ ] Remove post-hoc polynomial fitting claims
- [ ] Test true generalization capability

### 7. Error Handling & Logging
- [ ] Add try-catch blocks to all training functions
- [ ] Implement comprehensive logging
- [ ] Save intermediate results
- [ ] Add progress indicators

## ✅ PRIORITY 3 (POLISH FOR PUBLICATION)

### 8. Documentation
- [ ] Document all function parameters
- [ ] Add docstrings to all functions
- [ ] Create proper README with installation
- [ ] Add usage examples

### 9. Reproducibility  
- [ ] Pin exact package versions in Project.toml
- [ ] Add random seed controls throughout
- [ ] Create one-command reproduction script
- [ ] Test on clean environment

### 10. Code Quality
- [ ] Remove debug print statements
- [ ] Clean up commented code
- [ ] Consistent naming conventions
- [ ] Add unit tests for critical functions 