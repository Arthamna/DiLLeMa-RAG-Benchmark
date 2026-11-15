# Vendored PipeRAG runtime

This directory contains the minimum model-inference implementation required to
load and export the RETRO checkpoint used by this benchmark.

- Source: `https://github.com/amazon-science/piperag`
- Commit: `b6eb158`
- Upstream model lineage: `https://github.com/TobiasNorlund/retro` at `6945cdd`
- License: Apache License 2.0 (see `LICENSE`)

`modeling_retro_inference.py` is the upstream inference model with only import
paths adjusted so it can live in a Python package. Benchmark-specific service,
retrieval scheduling, index construction, and reporting live under the main
package; the runtime never imports from the local `reference/` checkout.
