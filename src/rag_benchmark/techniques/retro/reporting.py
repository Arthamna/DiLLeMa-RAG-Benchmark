from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ...shared.metrics import write_json


def write_pareto(output: Path, rows: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "pareto.json", rows)
    fieldnames = sorted({key for row in rows for key in row}) or ["technique"]
    with (output / "pareto.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("The paper extra with matplotlib is required to render pareto.png") from exc
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for technique in sorted({str(row.get("technique")) for row in rows}):
        group = [row for row in rows if row.get("technique") == technique]
        axis.scatter(
            [float(row.get("latency_ms", 0.0)) for row in group],
            [float(row.get("perplexity", 0.0)) for row in group],
            label=technique,
        )
    axis.set_xlabel("Mean sequence latency (ms)")
    axis.set_ylabel("Token-weighted perplexity")
    axis.set_title("RETRO / PipeRAG quality–latency trade-off")
    axis.grid(True, alpha=0.25)
    if rows:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "pareto.png", dpi=160)
    plt.close(figure)
