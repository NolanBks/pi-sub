# OpenPI π0.5 RoboCasa365 云服务器搭建指南（Conda，无 sudo）

本指南用于当前项目的 `pi05_robocasa365_subtask` 配置：从 `pi05_base` checkpoint 开始，在带 LeRobot `language_persistent` 子任务标注的 RoboCasa365 数据上做 JAX 微调。

## 1. 前提和服务器检查

建议使用 Ubuntu 22.04、单机 NVIDIA GPU、Python 3.11。当前配置是全参数微调，建议 A100/H100 80GB；显存低于约 70GB 时不要直接正式训练。

```bash
nvidia-smi
conda --version
git --version
git lfs --version
```

JAX CUDA 12 需要 Linux NVIDIA 驱动至少为 525。

## 2. 创建 Conda 环境

```bash
conda create -y -n openpi-pi05 python=3.11 pip setuptools wheel
conda activate openpi-pi05

python --version
which python
```

应显示 Python 3.11，且路径位于 `envs/openpi-pi05/bin/python`。

## 3. 准备项目和缓存

以下均为示例路径，请替换为实际路径：

```bash
cd /data/openpi-subtask
git submodule update --init --recursive

mkdir -p /data/openpi_cache
mkdir -p /data/huggingface_cache
mkdir -p /data/openpi_checkpoints
mkdir -p /data/cache

export OPENPI_DATA_HOME=/data/openpi_cache
export HF_HOME=/data/huggingface_cache
export XDG_CACHE_HOME=/data/cache
```

如需长期生效，将三个 `export` 写入自己的 `~/.bashrc`。

## 4. 安装 uv，并保持 Conda 为运行环境

将 uv 安装为用户级工具，不需要 sudo：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

激活 Conda 环境后，在项目目录同步依赖：

```bash
conda activate openpi-pi05
cd /data/openpi-subtask

GIT_LFS_SKIP_SMUDGE=1 uv sync --active --frozen
```

若 lockfile 与项目文件不匹配：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --active
```

验证 uv 使用的是 Conda 而不是项目下的 `.venv`：

```bash
uv run --active python -c "import sys; print(sys.executable)"
```

输出应包含 `envs/openpi-pi05`。

## 5. 验证 JAX 和 GPU

```bash
uv run --active python - <<'PY'
import jax
import torch

print("JAX devices:", jax.devices())
print("Torch CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

JAX 必须显示 `CudaDevice`。若只显示 `CpuDevice`，先解决驱动或 JAX CUDA 问题。

## 6. 登录 Hugging Face 并设置数据集

私有数据集需要登录：

```bash
uv run --active huggingface-cli login
export ROBOCASA_REPO="YOUR_ORG/YOUR_ROBOCASA365_ANNOTATED_DATASET"
```

## 7. 必须先验证 LeRobot 数据读取

当前代码从 `language_persistent` 中选择当前帧有效的 `style="subtask"` 行，并为跨子任务的 action chunk 生成 `action_loss_mask`。

```bash
uv run --active python - <<'PY'
import dataclasses
import os

from openpi.training import config
from openpi.training import data_loader

repo_id = os.environ["ROBOCASA_REPO"]
cfg = config.get_config("pi05_robocasa365_subtask")
cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=repo_id))
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

dataset = data_loader.create_torch_dataset(
    data_cfg,
    action_horizon=cfg.model.action_horizon,
    model_config=cfg.model,
)
sample = dataset[0]

print("keys:", sorted(sample.keys()))
print("timestamp:", sample.get("timestamp"))
print("prompt:", sample.get("prompt"))
print("language_persistent:", sample.get("language_persistent"))
print("action shape:", sample["action"].shape)
PY
```

样本至少应有：

```text
timestamp
action
language_persistent
observation.images.robot0_agentview_left
observation.images.robot0_eye_in_hand
observation.images.robot0_agentview_right
observation.state
```

并且 `language_persistent` 至少有一条非空记录：

```python
{
    "style": "subtask",
    "content": "reach for the fridge handle",
    "timestamp": 0.0,
}
```

第一条子任务必须从 episode 第 0 帧或 `timestamp=0.0` 起生效；时间戳应递增；当前配置要求数据集 FPS 为 20。

## 8. LeRobot 版本兼容警告

当前项目锁定原生 OpenPI 的旧 LeRobot revision。虽然代码已兼容新旧 import 路径，但旧 reader 未必能读取带 `language_persistent/language_events` 的新 v3.x 数据集。

如果第 7 步出现数据集版本不支持、找不到 `language_persistent`、v3.x 无法读取，或 parquet schema/import 错误，请停止，不要直接运行：

```bash
pip install -U lerobot
```

新 LeRobot 通常要求 Python 3.12 和 NumPy 2.x，而当前项目使用 Python 3.11、NumPy `<2.0` 与 Transformers 4.53。直接升级会破坏此环境。此时保存数据集 `meta/info.json` 的 `codebase_version` 和完整报错，再进行受控依赖迁移或数据预转换。

## 9. 计算 normalization statistics

仅当第 7 步成功后执行：

```bash
uv run --active scripts/compute_norm_stats.py \
  --config-name=pi05_robocasa365_subtask \
  --repo-id="$ROBOCASA_REPO"
```

该步骤为 `state` 和 `actions` 生成训练需要的归一化统计量。

## 10. 2 step smoke test

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run --active scripts/train.py pi05_robocasa365_subtask \
  --data.repo-id="$ROBOCASA_REPO" \
  --checkpoint-base-dir=/data/openpi_checkpoints \
  --exp-name=robocasa365_smoke \
  --num-train-steps=2 \
  --batch-size=2 \
  --no-wandb-enabled \
  --overwrite
```

监控显存：

```bash
watch -n 1 nvidia-smi
```

确认 `pi05_base` 能加载、三路图像和子任务正常读取、loss 非 NaN，且 checkpoint 可写入。

## 11. 正式训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run --active scripts/train.py pi05_robocasa365_subtask \
  --data.repo-id="$ROBOCASA_REPO" \
  --checkpoint-base-dir=/data/openpi_checkpoints \
  --exp-name=robocasa365_subtask_v1 \
  --overwrite
```

建议通过 tmux 保持任务：

```bash
tmux new -s pi05_train
```

训练启动后按 `Ctrl-b`、`d` 分离会话；重新连接：

```bash
tmux attach -t pi05_train
```

## 12. 单机多卡与恢复训练

4 卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run --active scripts/train.py pi05_robocasa365_subtask \
  --data.repo-id="$ROBOCASA_REPO" \
  --checkpoint-base-dir=/data/openpi_checkpoints \
  --exp-name=robocasa365_subtask_v1 \
  --fsdp-devices=4 \
  --overwrite
```

恢复训练时不要同时使用 `--overwrite`：

```bash
uv run --active scripts/train.py pi05_robocasa365_subtask \
  --data.repo-id="$ROBOCASA_REPO" \
  --checkpoint-base-dir=/data/openpi_checkpoints \
  --exp-name=robocasa365_subtask_v1 \
  --resume
```

当前代码支持单机多卡，不支持多节点。默认全局 batch size 为 64，必须能被实际训练设备数合理分配。

## 13. 推荐顺序

```text
创建 Conda 环境
    ↓
安装项目依赖
    ↓
确认 JAX 使用 CudaDevice
    ↓
读取一个带 language_persistent 的样本
    ↓
确认数据版本和 FPS
    ↓
计算 normalization statistics
    ↓
2 step smoke test
    ↓
正式训练
    ↓
checkpoint 恢复测试
```

