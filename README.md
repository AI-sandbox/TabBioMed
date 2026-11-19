<p><img src="docs/assets/logo.png" width="128" height="128" alt="TabBioMed logo"></p>

# TabBioMed: A Large-Scale Benchmark<br>for Biomedical Tabular Learning

A living benchmark for biomedical tabular learning.

[📄 Read the paper on arXiv](https://arxiv.org/abs/2611.23779) · [🏆 Live leaderboard](https://huggingface.co/spaces/TabBioMed/leaderboard) · [Datasets](https://huggingface.co/datasets/TabBioMed/TabBioMed)

The benchmark includes 94 datasets across eight biomedical domains and 23 model implementations. The [dataset catalogue](docs/datasets.md) lists the paper scope. Datasets retain their original licenses and attribution requirements.

## Installation

Python 3.12 is the tested runtime. Model and source-curation dependencies are optional. To install:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Install the groups needed for your models, for example `uv pip install -e ".[trees,neural]"`.

| Group | Models |
|---|---|
| Core | LinearModel, LassoModel, RidgeModel, ElasticNet, KNN, SVM, DecisionTree, RandomForest, MLP |
| `trees` | XGBoost, LightGBM, CatBoost |
| `neural` | RealMLP, TabM |
| `pfns` | TabPFNv2.5, RealTabPFNv2.5 |
| `tabicl`, `tabdpt`, `hyperfast`, `iltm` | Corresponding foundation model |
| `kernels` | NeoLsSVM |
| `xrfm` | xRFM |

Use a separate environment for TabPFN-Wide:

```bash
uv venv --python 3.12 .venv-wide
uv pip install --python .venv-wide/bin/python -e ".[tabpfn-wide]"
```

## Data

Official benchmark evaluations use the released datasets from [TabBioMed/TabBioMed](https://huggingface.co/datasets/TabBioMed/TabBioMed), at the pinned release revision. Local copies must match the released feature values, targets, row and column order, and splits.

Use `download` to fetch the benchmark data and `verify` to check that local datasets match the release:

```text
tabbiomed data download [DATASET ...] [--data-root PATH]
tabbiomed data verify [DATASET ...] [--data-root PATH]
```

Without specified dataset names, both commands process all benchmark datasets. Specify a name such as `pima_diabetes` to process only that dataset, and use `tabbiomed data list` to see available names.

Processed datasets are stored in `data/processed` by default. Use `--data-root PATH` to choose another directory, and pass the same path when downloading, verifying and evaluating datasets.

Downloaded files and NumPy conversions are also cached under `~/.cache/tabbiomed`. To relocate this cache, set `TBM_DATA_CACHE` before running a command, for example `export TBM_DATA_CACHE=/path/to/cache` in Bash. This setting does not change `--data-root`.

Raw curation code, including ESM embedding generation, is provided for reference and independent use. Running the benchmark does not require reconstructing datasets from raw sources. See [reference curation](docs/datasets.md#reference-curation).

## Evaluation

Each model and dataset evaluation uses the stored test split or all five stored outer folds. Validation selection includes Default and up to 100 sampled configurations within six hours for fixed splits or two hours per outer fold. iLTM uses its 25 recommended configurations. LinearModel and NeoLsSVM evaluate Default only. Recorded search trajectories support the 25/50/100/200 configuration comparison within the same selection budget.

```bash
tabbiomed run --dataset pima_diabetes --model LinearModel --output results/raw
```

Each fit runs in a fresh child process. Selection-stage process startup and serialization count against the selection budget. Failed fits are recorded and the controller continues while the budget allows.

Slurm jobs use a 24-hour limit per model and dataset pair, including all outer folds and selected final fits. Preparing jobs does not submit them. Choose partitions available on your cluster:

```bash
tabbiomed jobs prepare --directory jobs/pima --datasets pima_diabetes --models LinearModel --partition long
tabbiomed jobs submit --directory jobs/pima
tabbiomed jobs submit --directory jobs/pima --execute
```

The submission command prints its actions unless `--execute` is supplied. Select TabPFN-Wide explicitly and use its environment's Python when preparing its jobs.

## Results

Raw runs retain metrics, predictions, labels and incremental search histories. Create an explicit run manifest, check the exports, then install them:

```bash
tabbiomed results manifest --raw results/raw --output results/runs.json
tabbiomed results consolidate --manifest results/runs.json --output results/consolidated
tabbiomed results consolidate --manifest results/runs.json --output results/consolidated --execute
```

Consolidated results contain `report.json`, `timing.json`, test predictions and labels, and search trajectories for tunable models. Existing exports are preserved. See [evaluation and consolidation](docs/evaluation.md).

## Analysis and plots

Install the optional analysis dependencies:

```bash
uv pip install -e ".[analysis]"
```

Compute leaderboards and generate figures or tables from consolidated results:

```bash
tabbiomed analyze elo --results results/consolidated --output outputs/elo
tabbiomed analyze pca --results results/consolidated --output outputs/pca
tabbiomed analyze reproduce --profile paper --results results/consolidated --output outputs/paper
```

By default, analyses download the pinned published results into `data/results/`. An explicit `--results` path or `TBM_RESULTS` selects a local collection. The paper profile preserves the published model comparisons and figure settings. See [analysis and reproduction](docs/analysis.md).

## Add a model

Implement the model interface, register it with its search space and task support, then pass the registration module through `--model-module` when evaluating, preparing jobs or consolidating. GPU resource requirements are part of model registration. See [the model interface](docs/models.md).

## Development

Install the analysis extra and run the tests. Model-specific tests run when their optional dependencies are installed.

```bash
uv pip install -e ".[analysis]"
python -m unittest discover -s tests -v
```

The tests exercise registration, evaluation and consolidation with small CPU models. They do not submit cluster jobs or download model checkpoints.

## Citation and license

If you use TabBioMed, please cite:

```bibtex
@inproceedings{tabbiomed2026,
  title     = {TabBioMed: A Large-Scale Benchmark for Biomedical Tabular Learning},
  author    = {Mateo-Bernad{\'o}, Pau and Polapragada, Saivenkata Nagavyjayanthi and Bonet, David and Arbiol Rakuljic, Pol and Mogas Pladevall, Laia and Comajoan Cara, Mar{\c{c}}al and Gonzalez-Ferrer, Jesus and Abante, Jordi and Mas Montserrat, Daniel and Haussler, David and Ioannidis, Alexander G.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

If you use individual datasets, cite their original authors as well. Code is licensed under [Apache-2.0](LICENSE). Dataset and model-checkpoint licenses remain separate.
