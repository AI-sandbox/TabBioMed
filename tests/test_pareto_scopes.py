import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from tabbiomed.analysis.figures import pareto


class ParetoScopes(unittest.TestCase):
    def test_each_scope_can_generate_its_own_manifest(self):
        config = pareto.load_benchmark_models_config()
        models = config.expected_models_regression
        settings = json.loads((pareto.CONFIG / "benchmark.json").read_text())["models"]
        default_only = {m for m, spec in settings.items() if not spec["hpo_enabled"]}
        competitors = pareto.expected_competitors(models, default_only)
        datasets = {f"task_{i}": {"task": "binary", "n_folds": 0} for i in range(94)}
        plan = {
            "datasets": datasets,
            "models": {m: {"hpo_enabled": m not in default_only} for m in models},
        }
        elo = pd.DataFrame({"competitor": competitors, "elo": 1000.0})
        metrics = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "competitor": competitor,
                    "model": competitor.split("::")[0],
                    "variant": competitor.split("::")[1],
                    "task": "binary",
                    "fold": 0,
                    "metric_value": 0.5,
                    "imputed": False,
                }
                for dataset in datasets
                for competitor in competitors
            ]
        )
        selections = [{"model": m, "dataset": d} for m in models for d in datasets]
        for scope in ("main", "domains", "regimes"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                elo_dir = root / "elo"
                (elo_dir / "rf_imputed").mkdir(parents=True)
                elo.to_csv(elo_dir / "rf_imputed/elo_all_entries.csv", index=False)
                metrics.to_csv(elo_dir / "rf_imputed/metric_inputs.csv", index=False)
                (elo_dir / "selected_reports.json").write_text(json.dumps(selections))
                scopes = [
                    (
                        "domain:sc",
                        "Single-cell",
                        list(datasets),
                        "per_domain",
                        "pareto_scatter_sc",
                    ),
                    (
                        "regime:q1",
                        "Q1",
                        list(datasets),
                        "per_regime",
                        "pareto_scatter_q1",
                    ),
                ]

                def render(summaries, config, output, stem="pareto_scatter", **kwargs):
                    (output / f"{stem}.pdf").write_bytes(b"rendered figure")

                argv = ["pareto", "--output", str(root / "plots"), "--scopes", scope]
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(pareto, "_OUTPUT_ROOT", root),
                    patch.object(pareto, "ensure_results"),
                    patch.object(pareto, "load_benchmark", return_value=plan),
                    patch.object(
                        pareto,
                        "collect_timings",
                        return_value=(
                            pd.DataFrame({"dataset": list(datasets)}),
                            pd.DataFrame({"search_clock_status": []}),
                        ),
                    ),
                    patch.object(
                        pareto,
                        "make_summaries",
                        return_value={clock: pd.DataFrame() for clock in pareto.CLOCKS},
                    ),
                    patch.object(pareto, "scope_datasets", return_value=scopes),
                    patch.object(pareto, "render", side_effect=render),
                ):
                    pareto.main()
                manifest = json.loads((root / "plots/manifest.json").read_text())
                self.assertEqual("figure_sha256" in manifest, scope == "main")
                self.assertEqual(
                    len(manifest["subset_figures"]), 0 if scope == "main" else 1
                )
                self.assertEqual(
                    (root / "plots/pareto_scatter.pdf").exists(), scope == "main"
                )

    def test_domain_scopes_do_not_read_regime_files(self):
        plan = {"datasets": {"example": {}}}
        with (
            patch.object(
                pareto,
                "analysis_dataset_metadata",
                return_value={"label": "Single-cell"},
            ),
            patch.object(pareto, "read_numeric_csv") as read,
        ):
            scopes = pareto.scope_datasets(plan, include_regimes=False)
        read.assert_not_called()
        self.assertTrue(all(scope[0].startswith("domain:") for scope in scopes))


if __name__ == "__main__":
    unittest.main()
