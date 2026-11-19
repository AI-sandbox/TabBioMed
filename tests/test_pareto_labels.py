import unittest

import pandas as pd

from tabbiomed.analysis.pareto_style import Panel, default_only_models, front_label


class ParetoLabels(unittest.TestCase):
    def test_default_only_model_uses_its_name(self):
        panel = Panel(
            pd.DataFrame(
                [
                    dict(model="OnlyDefault", variant="default"),
                    dict(model="Tunable", variant="default"),
                    dict(model="Tunable", variant="tuned"),
                ]
            ),
            "cost",
            "Cost",
        )
        models = default_only_models([panel])
        self.assertEqual(models, {"OnlyDefault"})
        self.assertEqual(front_label("OnlyDefault", "default", models), "OnlyDefault")
        self.assertEqual(front_label("Tunable", "default", models), "Tunable-D")
        self.assertEqual(front_label("Tunable", "tuned", models), "Tunable-T")

    def test_variants_are_checked_across_all_panels_and_points(self):
        panels = [
            Panel(
                pd.DataFrame(
                    [
                        dict(model="Example", variant="default", is_pareto=True),
                    ]
                ),
                "cost",
                "Cost",
            ),
            Panel(
                pd.DataFrame(
                    [
                        dict(model="Example", variant="tuned", is_pareto=False),
                    ]
                ),
                "cost",
                "Cost",
            ),
        ]
        models = default_only_models(panels)
        self.assertEqual(models, set())
        self.assertEqual(front_label("Example", "default", models), "Example-D")


if __name__ == "__main__":
    unittest.main()
