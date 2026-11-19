# Evaluation and results

Run a complete model/dataset pair with the installed environment:

```bash
tabbiomed run --model RandomForest --dataset pima_diabetes --data-root data/processed --output results/raw
```

Each run writes a separate `results/raw/<Model>/<Dataset>/run_<timestamp>/`
directory. It contains the report, predictions, test labels, incremental HPO
history and a `manifest.json` that selects that run for consolidation.

## Validation and refitting

Selection compares Default and up to 100 attempted HPO configurations. Default
validation is charged to the same six-hour fixed-split budget or two-hour budget
for each stored outer CV fold. It does not consume a separate sampled-attempt
slot. iLTM uses its 25 recommended configurations. Finite spaces may stop earlier.

For CV, selection uses the mean over three requested inner folds, reduced to two
when class or group support requires it, or the recorded stored holdout when
neither is feasible. Preprocessing is fitted independently on each inner
training partition. All inner primary metrics must succeed, and every stored
outer test fold must be present for an aggregate test metric.

The binary-category shortcut uses `0` and `1` for known values and `-1` for
missing or unseen values.

The official selection is frozen at its attempt cap. Collection may continue to
200 attempts within the same wall-clock allowance. Saved validation selections
at caps 25, 50, 100 and 200 receive test evaluations after selection. Identical
selected configurations share their evaluation. An unreached cap uses only its
available in-budget prefix. Test scores never select configurations.

Every fit runs in a fresh Python process. Interpreter startup and serialization
for validation configurations consume the selection wall clock. Failed attempts
consume their original slots and elapsed time. Native failures terminate the
fit process, and the controller continues while budget remains. Default test
fits and selected test refits are outside the selection clock, within the job's
24-hour limit. Test refits merge train and validation when the model permits it,
and preserve the stored holdout when fitting uses validation.

LinearModel and NeoLsSVM use StandardScaler and Default evaluation only. Training
and validation prediction times are recorded separately from test prediction.
Published Tuned training costs include measured selection time and selected
final fitting/validation prediction. No model checkpoints are retained.

## Slurm

Prepare jobs without submitting:

```bash
tabbiomed jobs prepare --directory jobs/run1 --models RandomForest TabM --datasets pima_diabetes ESOL --partition cpu --gpu-partition gpu --gpu-type a5500
```

Every array task evaluates one complete model/dataset pair, including all outer
folds and selected refits, with 24 hours, 10 CPUs and 128 GB RAM. GPU models
request one GPU. Partition and GPU type are cluster-specific. Omitting models
selects the registered benchmark models except TabPFN-Wide. Select TabPFN-Wide
explicitly using its separate environment, or prepare its plan with `--python`
pointing to that environment's interpreter. Additional models declare GPU needs
through registration. `--throttle N` limits concurrent tasks per array.

Review the submission commands, then submit explicitly:

```bash
tabbiomed jobs submit --directory jobs/run1
tabbiomed jobs submit --directory jobs/run1 --execute
```

The submission receipt records accepted arrays. Repeating submission skips
already submitted arrays and submits only an unsubmitted array. Preparation
and preflight never fit models or submit jobs. Install the package on compute
nodes, and make its environment and chosen data/output directories accessible.

## Consolidation

Use the manifest written by an individual run, or collect a settled collection:

```bash
tabbiomed results manifest --raw results/raw --output runs.json
tabbiomed results consolidate --manifest runs.json --output results/consolidated
tabbiomed results consolidate --manifest runs.json --output results/consolidated --execute
```

Manifest generation rejects active runs and multiple runs for one model/dataset
pair. Select the intended report explicitly when several runs exist:

```json
{
  "schema_version": 1,
  "evaluations": [
    {
      "report": "raw/RandomForest/pima_diabetes/run_example/report.json",
      "data_root": "data/processed/pima_diabetes"
    }
  ]
}
```

Paths are relative to the manifest, or absolute. Each entry requires `report`.
`data_root` selects the directory containing `X.npy`, `y.npy`, metadata and
stored indices, and otherwise uses the run's recorded input directory.
Optional `model` and `dataset` values must match the report. `scope` defaults to
`main`, and a named scope is stored below its corresponding Main evaluation.
`hardware` may contain `name` and `total_memory`. Explicit failure evidence may
use `failures` by `default`/`best`, or `fold_failures` by outer fold, plus
`search_failure` for a search-wide outcome. Failure annotations accept only
`reason` and `memory_limit_gib`. Successful test results cannot be
annotated as failed. No phase annotations are exported.

Consolidation validates input identities, stored test labels, selected parameters,
validation scores, budgets and prediction references. It creates portable
`<Model>/<Dataset>/report.json`, `timing.json`, predictions, labels and HPO
sidecars. Timing split identities describe row order, split membership and groups,
independently of NumPy file serialization. Reports keep failed outcomes explicitly. A scheduler-stopped run retains all
stored outer-fold entries and any completed evaluations. Add confirmed
`time_limit` failure evidence to its manifest before publishing unfinished
evaluations. An incomplete CV never receives a partial aggregate test score.
Existing evaluation destinations are never overwritten, and no fitting or job
submission occurs during consolidation. Add `--log-dir PATH` to save optional
consolidation logs. Omit it to write only the results. The log directory may be
inside the results root, but cannot overwrite inputs or evaluation folders.

For an additional model registered by a Python module, pass the same
`--model-module your_module` to `run`, `jobs prepare`, `results manifest` and
`results consolidate`. The job manifest forwards it to the evaluation process.
See [models.md](models.md) for the model interface and registration.

For external PCA, the [`pca500` profile](pca.md) prepares the paper's matched 500-component experiment. Its raw outputs and consolidated scope are separate from Main.
