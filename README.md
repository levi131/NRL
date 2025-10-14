
# NativeRL

<!-- Language switch badges -->
[![English](https://img.shields.io/badge/English-%F0%9F%87%BA%F0%9F%87%B8-blue)](README.md) [![中文](https://img.shields.io/badge/中文-%F0%9F%87%A8%F0%9F%87%B3-red)](README.zh-CN.md)

NativeRL — a lightweight reinforcement learning framework built on native PyTorch.

Overview

NativeRL focuses on providing concise, modular tools for RL research and engineering without depending on large end-to-end training/inference frameworks. It builds distributed and inference capabilities directly on PyTorch primitives and offers flexible parallel strategies and mixed training/inference workflows.

Canonical sections in this README (both language files are parallel):

- Key features
- Project layout
- Installation
- Quick start (examples)
- Distributed runs
- Contributing
- License & Contact

Key features

- Native PyTorch implementation (no heavy external infra).
- Implementations for common RL algorithms: DPO, PPO, GRPO, etc.
- Distributed support via PyTorch DTensor (data/model/tensor parallel, sharding).
- Training and inference can run in the same process to support online evaluation.
- Modular design: swap/extend algorithms, policies, and backends.

Project layout (high level)

- `nrl/` — core package (entry points, algorithms, training, distribution utilities)
- `examples/` — runnable examples and configs (e.g. `examples/ardf`)
- `scripts/`, `tests/` — utilities and tests

Installation

Prerequisites: Python and a matching PyTorch build (GPU support if you plan to use GPUs).

```bash
# Use a virtual environment
pip install -r requirements.txt
```

Quick start — run the `ardf` example

The `examples/ardf/run.sh` contains the run command:

```bash
python3 nrl/entry.py examples/ardf/config.py
```

This runs the repository entry point with the `examples/ardf/config.py` configuration. To run a different example/update the config path accordingly.

Distributed runs

For multi-process distributed runs, use PyTorch's launcher. Example (N processes on a single node):

```bash
python -m torch.distributed.run --nproc_per_node=<N> nrl/entry.py examples/ardf/config.py
```

Contributing

- Open an Issue to discuss major changes.
- Add/update examples or tests when adding features.
- Keep code style consistent and provide benchmarks for performance-sensitive changes.

License & Contact

See `LICENSE` in the repository root. For questions, open an Issue or contact the maintainer.
