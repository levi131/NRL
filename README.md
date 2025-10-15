

# NativeRL

<!-- In-page language links (stay on this page) -->
[![English](https://img.shields.io/badge/English-%F0%9F%87%BA%F0%9F%87%B8-blue)](#english) [![中文](https://img.shields.io/badge/中文-%F0%9F%87%A8%F0%9F%87%B3-red)](#中文)

Click a language to jump to that section on this page (no navigation to a different file).

---

<a id="english"></a>

# English

NativeRL — a lightweight reinforcement learning framework built on native PyTorch.

## Overview

NativeRL focuses on providing concise, modular tools for RL research and engineering without depending on large end-to-end training/inference frameworks. It builds distributed and inference capabilities directly on PyTorch primitives and offers flexible parallel strategies and mixed training/inference workflows.

## Key features

- Native PyTorch implementation (no heavy external infra).
- Implementations for common RL algorithms: DPO, PPO, GRPO, etc.
- Distributed support via PyTorch DTensor (data/model/tensor parallel, sharding).
- Training and inference can run in the same process to support online evaluation.
- Modular design: swap/extend algorithms, policies, and backends.

## Project layout (high level)

- `nrl/` — core package (entry points, algorithms, training, distribution utilities)
- `examples/` — runnable examples and configs (e.g. `examples/ardf`)
- `scripts/`, `tests/` — utilities and tests

## Installation

Prerequisites: Python and a matching PyTorch build (GPU support if you plan to use GPUs).

```bash
# Use a virtual environment
pip install -r requirements.txt
```

## Quick start — run the `ardf` example

The `examples/ardf/run.sh` contains the run command:

```bash
python3 nrl/entry.py examples/ardf/config.py
```

This runs the repository entry point with the `examples/ardf/config.py` configuration. To run a different example/update the config path accordingly.

## Distributed runs

For multi-process distributed runs, use PyTorch's launcher. Example (N processes on a single node):

```bash
python -m torch.distributed.run --nproc_per_node=<N> nrl/entry.py examples/ardf/config.py
```

## Contributing

- Open an Issue to discuss major changes.
- Add/update examples or tests when adding features.
- Keep code style consistent and provide benchmarks for performance-sensitive changes.

## License & Contact

See `LICENSE` in the repository root. For questions, open an Issue or contact the maintainer.

---

<a id="中文"></a>

# 中文

NativeRL — 基于原生 PyTorch 的轻量级强化学习框架。

## 概览

NativeRL 专注于为强化学习研究与工程提供简洁且模块化的工具，不依赖大型端到端训练/推理平台。项目直接基于 PyTorch 原语构建分布式与推理能力，支持灵活的并行策略与混合训练/推理工作流。

## 主要特性

- 原生 PyTorch 实现（不依赖大型外部 infra）。
- 常见算法实现：DPO、PPO、GRPO 等。
- 基于 PyTorch DTensor 的分布式支持（数据/模型/张量并行、切分等）。
- 支持在同一进程中进行训练与推理，便于在线评估。
- 模块化设计：便于替换或扩展算法、策略与后端。

## 项目结构（高层）

- `nrl/` — 核心包（入口、算法、训练、分布式工具）
- `examples/` — 可运行示例与配置（例如 `examples/ardf`）
- `scripts/`, `tests/` — 实用脚本与测试

## 安装

前提：已安装 Python 与合适的 PyTorch（如需使用 GPU，请安装对应的 CUDA 支持版本）。

```bash
# 使用虚拟环境
pip install -r requirements.txt
```

## 快速开始 — 运行 `ardf` 示例

`examples/ardf/run.sh` 中包含以下运行命令：

```bash
python3 nrl/entry.py examples/ardf/config.py
```

该命令会以 `examples/ardf/config.py` 作为配置启动仓库入口脚本 `nrl/entry.py`。如需运行其他示例，请替换配置路径。

## 分布式运行（示例）

使用 PyTorch 分布式启动器进行多进程训练，例如（单机 N 进程）：

```bash
python -m torch.distributed.run --nproc_per_node=<N> nrl/entry.py examples/ardf/config.py
```

## 贡献

- 在提交大改动前建议先通过 Issue 讨论设计。
- 增加功能时请提供或更新示例和测试用例。
- 对性能相关改动建议附带基准或对比数据。

## 许可与联系方式

详见仓库根目录下的 `LICENSE` 文件。如有问题，请通过 Issue 联系维护者。
