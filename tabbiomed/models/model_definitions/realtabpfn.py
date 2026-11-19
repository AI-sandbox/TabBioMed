"""Real-data TabPFN checkpoints with the shared TabPFN interface."""

from typing import Dict, Any

from .tabpfn_v2_5 import TabPFN

REALTABPFN_DIR = None
REALTABPFN_CHECKPOINTS = [
    "tabpfn-v2.5-classifier-v2.5_default.ckpt",
    "tabpfn-v2.5-regressor-v2.5_real.ckpt",
]


class RealTabPFN(TabPFN):
    """Use real-data default checkpoints with the shared TabPFN interface."""

    @classmethod
    def get_base_params(cls, args):
        return {
            **super().get_base_params(args),
            "zip_model_path": list(REALTABPFN_CHECKPOINTS),
            "custom_model_dir": REALTABPFN_DIR,
        }

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Keep the baseline checkpoints independent of the HPO choices."""
        return {
            **TabPFN.get_default_params(),
            "zip_model_path": list(REALTABPFN_CHECKPOINTS),
        }
