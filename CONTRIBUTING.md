# Contributing to TabBioMed

Open a [GitHub issue](https://github.com/AI-sandbox/TabBioMed/issues) to report a bug or suggest a change.

## Report a bug

Include the TabBioMed version or commit, the command you ran, and what you expected to happen. Share the error message or relevant log excerpt and the smallest set of steps that reproduces the problem. If it affects an evaluation, include the model and dataset names.

## Suggest a change

Describe what you were trying to do, where you got stuck, and what you’d like to change. A short example helps us understand the request.

## Suggest a dataset

Open a GitHub issue with a link to the original dataset and include its license. Briefly describe the biological context and prediction task, and indicate which of TabBioMed’s biomedical domains (or new domain) the dataset fits. Mention any existing splits to preserve or groups that should stay together.

If you’re preparing curation code, see [Adding datasets](docs/datasets.md#adding-datasets) for what to include.

## Suggest a model

Open a GitHub issue with a link to the model’s paper or implementation. Include the model version, the prediction tasks it supports, and any hardware requirements.

If you’re adding an implementation, see the [model guide](docs/models.md) for how to register the model and run it with `--model-module`. Include the command you used to test it on a benchmark dataset.
