# Analysis and reproduction

The optional analysis module computes leaderboards, statistical comparisons and the figures and tables used in the paper. It consumes consolidated `report.json`, `timing.json`, test predictions and labels, and `hpo/` trajectories.

```bash
uv pip install -e ".[analysis]"
tabbiomed analyze --help
```

## Commands

```bash
tabbiomed analyze elo --results results/consolidated --output outputs/elo
tabbiomed analyze cd --results results/consolidated --output outputs/cd
tabbiomed analyze pca --results results/consolidated --output outputs/pca
tabbiomed analyze reproduce --profile paper --results results/consolidated --output outputs/paper
```

Each command accepts `--help`. Individual entry points remain available, for example `python -m tabbiomed.analysis.figures.elo` and `python -m tabbiomed.analysis.tables.model_summary`. The reproduction command runs the generators in dependency order. Paths are resolved from the working directory, and installed commands work outside the source checkout.

## Inputs

By default, results download from the pinned `TabBioMed/results` Hugging Face release into `data/results/`. Completed files are reused and rate-limited downloads resume. An explicit `--results` path or `TBM_RESULTS` selects a local consolidated collection.

`--plan` sets the dataset and model scope, and `--model-families` sets model families
and task eligibility across analysis commands. Default-only status comes from the
recorded HPO policy.

Dataset metadata, values and identities use the benchmark's shared data loader and pinned `TabBioMed/TabBioMed` release. Analyses such as metafeature extraction request the dataset values they need. Use `TBM_PROCESSED` for an existing processed NumPy directory or `TBM_PARQUET` for a downloaded Parquet snapshot. Set at most one of these variables. Scientific identity checks apply to either source. HF authentication uses `hf auth login` or `HF_TOKEN` when needed.

Outputs default to `outputs/`, configurable through `--output` or `TBM_OUTPUT`. `TBM_DATA_CACHE` controls the download and dataset-conversion cache. Matplotlib panels that specify Times New Roman require that font to reproduce the paper typography.

## Comparisons

The paper profile retains the 21-model all-task and 23-model classification comparisons. Its model scope, comparison groups, styles, table templates and entry-point mapping are packaged with the analysis module. Dataset identities come from the benchmark catalog.

Missing-result policies and Default fallback remain explicit in each analysis. The main Elo figure averages outer-CV metrics before comparisons and imputes missing entries with RF Default. Splitwise Elo and complete-case comparisons are available separately. Main analyses use Main evaluations. PCA compares `pca_500/` with Main, independently using Default when Tuned is unavailable. A100 substitution is limited to the dedicated A100 analysis. HPO analyses read the consolidated selections at 25, 50, 100 and 200 attempts.

Calculations live in `tabbiomed.analysis`, figure renderers in `analysis.figures`, table renderers in `analysis.tables`, and analyses producing related outputs in `analysis.studies`. These functions can also be used directly with new model results. Training metric computation remains in `tabbiomed.evaluator`.
