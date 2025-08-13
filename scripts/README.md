# Scripts and CLI

Use the `bin/mg` helper to run common tasks from the project root:

- mg tune: hyperparameter tuning (writes `checkpoints/hparam_search_results.csv`, `checkpoints/best_hparam_config.bson`)
- mg train: training with config/env overrides
- mg eval: evaluation on test set
- mg verify: parameter/activation/simulation checks
- mg results: regenerate final results table
- mg figs: regenerate figures
- mg fresh: fresh evaluation snapshot and figures

Config
- Optional config file at `config/config.toml` overrides defaults; environment variables still take precedence.

Examples
```bash
mg tune
TRAIN_SUBSET_SIZE=1500 TRAIN_SAMPLES=1000 mg train
mg eval
mg results
mg figs
``` 