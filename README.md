# NativeRL

NativeRL 是一个轻量级强化学习框架，直接基于原生 PyTorch 开发。它不是在 vllm、sglang、Megatron 等大型框架上做缝合或二次封装，而是从 PyTorch 原生构建分布式训练与推理能力，旨在为研究和工程中需要灵活并行策略与混合训练/推理流程的场景提供简洁、高效的工具。

## 主要特点

- 原生 PyTorch 实现：不依赖大型第三方训练或推理框架，更加专注于解决强化学习这一场景下的 infra 问题。
- 支持常见算法：目前包含 DPO（Direct Preference Optimization）、PPO（Proximal Policy Optimization）、GRPO 等常见强化学习算法。
- 分布式支持：基于 PyTorch DTensor 体系实现分布式能力，支持不同的分布式并行策略（数据并行、模型并行、张量分割等）。
- 训练/推理同进程切换：训练与推理两个阶段可以在同一进程中交替进行，便于实现在线评估、交互式学习和自适应训练策略。
- 可扩展与模块化：算法、策略、分布式后端等模块化设计，方便研究人员替换或新增组件。

## 设计哲学

- 简洁优先：保持代码风格贴近 PyTorch 原生 API，降低学习成本。
- 可控并行：提供多种并行策略选择，而不是强制一种实现方式，便于针对任务调优性能与内存使用。
- 研究与工程并重：既支持快速原型验证，也考虑实际部署时的推理效率与可靠性。

## 目录结构（示例）

- configs/         - 训练与分布式配置
- envs/            - 环境与交互封装（Gym/自定义环境）
- algorithms/      - DPO、PPO、GRPO 等算法实现
- trainers/        - 训练循环、优化器、策略管理
- dist/            - 基于 DTensor 的分布式工具与策略
- examples/        - 简单的训练/评估示例

（实际仓库目录可能略有不同，请以源代码为准）

## 快速开始

以下为快速体验步骤（假设你已安装 PyTorch 并具备合适的 GPU/分布式环境）：

1. 安装依赖

```bash
# 建议在虚拟环境中执行
pip install -r requirements.txt
```

2. 单机快速训练（示例）

```bash
python examples/train.py --config configs/ppo_cartpole.yaml
```

3. 分布式训练（基于 DTensor 的简单示例）

```bash
# 以两个进程为例（仅示意）
python -m torch.distributed.run --nproc_per_node=2 examples/train.py --config configs/ppo_distributed.yaml
```

4. 在同一进程中交替进行训练与推理

NativeRL 支持在训练主循环中周期性触发推理/评估，或在推理期间短暂停止训练，二者在同一进程内完成。示例用法：

```python
# 在 trainer 中使用
for epoch in range(start, max_epochs):
    trainer.train_one_epoch()
    if epoch % cfg.eval_interval == 0:
        # 在同一进程中进行推理或在线评估
        trainer.run_evaluation()
```

更多细节请参考 `examples/` 中的脚本。

## 支持的算法（简述）

- DPO：直接基于偏好数据优化策略，适合有偏好信号的强化学习任务。
- PPO：一种常用的策略梯度算法，兼顾稳定性和样本效率。
- GRPO：一种具有更强鲁棒性或特定约束下的策略优化方法（具体实现参见源码）。

## 分布式与并行策略

NativeRL 基于 PyTorch DTensor 实现分布式能力，支持：

- 数据并行（DataParallel / DistributedDataParallel）
- 张量分割与张量并行（通过 DTensor 划分张量维度）
- 混合并行策略：在模型不同部分采用不同并行方式

用户可以通过配置文件选择并行策略、设备拓扑以及 DTensor 切分规则。

## 使用场景

- 需要在训练过程里进行在线评估或人类偏好收集的 RL 任务。
- 需自定义并行策略以适配特殊硬件拓扑的工程项目。
- 学术研究：需要直接修改原生 PyTorch 代码以验证新假设或算法。

## 贡献指南

欢迎提交 Issue 和 Pull Request。建议先在 Issue 中讨论大改动或新特性。提交 PR 前请确保：

- 添加/更新对应的示例或测试用例。
- 保持代码风格与现有仓库一致。
- 对于性能相关改动，提供基准或对比数据。

## 许可

请参考仓库根目录下的 LICENSE 文件（如无则联系作者）。

## 联系方式

如有问题或建议，请通过仓库 Issue 或直接联系作者（levi131）。
