import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src/rag_benchmark/techniques/rankify/worker/src"),
)

