import os
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs"
DATA_CACHE = (
    Path(os.environ.get("TBM_DATA_CACHE", Path.home() / ".cache/tabbiomed"))
    .expanduser()
    .resolve()
)
PARQUET = (
    Path(os.environ["TBM_PARQUET"]).expanduser().resolve()
    if os.environ.get("TBM_PARQUET")
    else None
)
