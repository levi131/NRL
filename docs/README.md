````markdown


# NativeRL

Welcome — this repository includes an interactive docs site that defaults to English and can switch to 中文 in-place.

» Interactive docs (recommended): https://levi131.github.io/NativeRL/

If you want the same in-page language toggle on the repository homepage, enable GitHub Pages for this repository:

1. Go to your repository Settings → Pages.
2. Under Source select branch: `main` and folder: `/docs`.
3. Save — the site will be published at: https://levi131.github.io/NativeRL/

Once Pages is enabled the interactive docs (English default with a one-click 中文 toggle) are available at the URL above. The rest of this README remains available in the repo if you prefer to view it directly.

---

<!-- The original README content is kept below for convenience. Use the Pages site for the interactive experience. -->

# English

NativeRL — a lightweight reinforcement learning framework built on native PyTorch primitives.

## Overview

NativeRL focuses on providing modular, easy-to-extend tools for reinforcement learning research and engineering without relying on large end-to-end platforms. The project is built directly on PyTorch primitives and supports distributed and inference capabilities.

## Main features

- Native PyTorch implementation (no heavy infra).
- Common algorithms (PPO, DPO, GRPO, ...).
- Distributed support using PyTorch primitives and DTensor.
- Training + inference in the same process for online evaluation.
- Modular design for easy extension.

## Layout

- `nrl/` — core package (entry, algorithms, training, distributed tools)
- `examples/` — runnable examples and configs
- `scripts/`, `tests/` — helper scripts and tests

## Installation

Prerequisites: Python and a matching PyTorch build.

```bash
# create a virtualenv
pip install -r requirements.txt
```

## Quick start — run `ardf` example

```bash
python3 nrl/entry.py examples/ardf/config.py
```

## Distributed example

```bash
python -m torch.distributed.run --nproc_per_node=<N> nrl/entry.py examples/ardf/config.py
```

## Contributing

- Open an Issue to discuss large changes before implementation.
- Provide or update examples and tests when adding features.
- Include benchmarks for performance-sensitive changes.

## License

See the `LICENSE` file in the repository root. Use Issues for questions or contact the maintainers.

This project is licensed under the Apache License 2.0. See `LICENSE` for details.

````
