# README

这是一个使用 jax/flax 复现 [Diffusion Policy for Offline RL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) 算法的代码仓库，同时也是我首次尝试使用torch以外的技术构建较为复杂的强化学习系统。

## 安装

### 1. 安装常规依赖

请先安装以下依赖包：

- **[D4RL](https://github.com/rail-berkeley/d4rl)**
- **[OpenAI Gym](https://github.com/openai/gym)**
- **[NumPy](https://numpy.org/)**
- **[JAX](https://github.com/google/jax)**
- **[Flax](https://github.com/google/flax)**
- **[Optax](https://github.com/deepmind/optax)**
- **[Chex](https://github.com/deepmind/chex)**
- **[Distrax](https://github.com/deepmind/distrax)**
- **[Weights & Biases](https://wandb.ai/)**

可通过以下命令进行安装（根据实际环境调整）：
```bash
pip install d4rl gym numpy jax flax optax chex distrax wandb
```

### 2. 克隆 & 本地安装 jaxrl_m

请执行以下命令：
```bash
git clone https://github.com/dibyaghosh/jaxrl_m.git
cd jaxrl_m
pip install -e .
```
使用 `-e` 参数以开发模式安装，可以确保项目依赖库代码指向本地的 **jaxrl_m**。

## 运行

在项目根目录下运行：
```bash
python run_<algo_name>.py
```

## 文件结构

- `hyper_<name>.py`：存放默认超参数及调参配置  
- `util_<name>.py`：包含数据加载、模型等工具函数  
- `model_<name>.py`：定义网络结构  
- `algo_<name>.py`：定义 RL Agent 的核心逻辑，包括创建、更新与采样  
- `run_<name>.py`：程序运行入口  
- `xxx_test.py`：测试文件

## 训练结果

训练过程和结果可在 **[Weights & Biases](https://wandb.ai/)** 平台上查看。

## 性能报告

在 RTX 4060 游戏本上的测试数据：
- 训练速度：从 ~38 it/s 提升至 ~650 it/s，速度显著加快。
- GPU 使用率：从 ~20% 上升至 ~45%，增长相对有限。
- GPU 显存利用率：从 ~15% 上升至 ~70%，得到了更充分的利用。

## 致谢

- 感谢 **[JAX](https://github.com/google/jax)**、**[Flax](https://github.com/google/flax)**、**[Optax](https://github.com/deepmind/optax)**、**[Distrax](https://github.com/deepmind/distrax)** 等高质量深度学习库，其代码和文档均极为优雅。  
- 感谢 [jaxrl](https://github.com/ikostrikov/jaxrl)、[jaxrl2](https://github.com/ikostrikov/jaxrl2) 以及 **[jaxrl_m](https://github.com/dibyaghosh/jaxrl_m/tree/main)** 对 JAX/Flax 在强化学习领域应用所做出的杰出贡献。  
- 感谢 *Diffusion Policy for Offline RL* 的原作者，其坚实的算法在框架迁移后依然保持了极高的复现性。

---

若在使用过程中遇到任何问题，欢迎提交 issue 或 PR，共同完善该项目！祝你在强化学习研究中取得更多成果！