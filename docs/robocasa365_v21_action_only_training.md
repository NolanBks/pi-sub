# RoboCasa365 v2.1 无子任务标注训练命令

本文档适用于当前项目新增的配置：

~~~text
pi05_robocasa365_v21_action_only
~~~

该配置从 pi05_base 初始化，只训练 π0.5 的 flow-matching action objective：

~~~text
高层 task prompt + 三路图像 + 16-D state → 12-D action chunk
~~~

它不读取 language_persistent，不训练子任务 token，也不在推理时生成子任务。

## 前置条件

- 已激活 Conda 环境，例如 conda activate openpi-pi05。
- 已在项目目录执行 uv sync --active。
- JAX 已能识别 GPU：uv run --active python -c "import jax; print(jax.devices())"。
- 数据是 LeRobot v2.1 RoboCasa365 格式，具有 action、task_index、observation.state 和三路相机字段。

## 1. 设置数据集

~~~bash
cd /data/openpi-subtask
conda activate openpi-pi05

export OPENPI_DATA_HOME=/data/openpi_cache
export HF_HOME=/data/huggingface_cache
export ROBOCASA_REPO="YOUR_ORG/YOUR_ROBOCASA365_V21_DATASET"
~~~

私有 Hugging Face 数据集需要先登录：

~~~bash
uv run --active huggingface-cli login
~~~

## 2. 读取一个样本进行检查

~~~bash
uv run --active python - <<'PY'
import dataclasses
import os

from openpi.training import config
from openpi.training import data_loader

repo_id = os.environ["ROBOCASA_REPO"]
cfg = config.get_config("pi05_robocasa365_v21_action_only")
cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=repo_id))
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
dataset = data_loader.create_torch_dataset(
    data_cfg, action_horizon=cfg.model.action_horizon, model_config=cfg.model
)
sample = dataset[0]

print("keys:", sorted(sample.keys()))
print("prompt:", sample.get("prompt"))
print("state shape:", sample["observation.state"].shape)
print("action shape:", sample["action"].shape)
PY
~~~

预期 action shape 为：

~~~text
(50, 12)
~~~

这个配置不应要求或访问 language_persistent。

## 3. 计算 normalization statistics

~~~bash
uv run --active scripts/compute_norm_stats.py \
  --config-name=pi05_robocasa365_v21_action_only \
  --repo-id="$ROBOCASA_REPO"
~~~

## 4. 两步 smoke test

先运行极短训练，确认数据映射、pi05_base 下载和 checkpoint 写入均正常：

~~~bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run --active scripts/train.py pi05_robocasa365_v21_action_only \
  --data.repo-id="$ROBOCASA_REPO" \
  --checkpoint-base-dir=/data/openpi_checkpoints \
  --exp-name=robocasa365_v21_smoke \
  --num-train-steps=2 \
  --batch-size=2 \
  --no-wandb-enabled \
  --overwrite
~~~

监控显存：

~~~bash
watch -n 1 nvidia-smi
~~~

检查 loss 不是 NaN，且日志中没有 language_persistent 或 Subtask is required 错误。

## 5. 正式训练

~~~bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run --active scripts/train.py pi05_robocasa365_v21_action_only \
  --data.repo-id="$ROBOCASA_REPO" \
  --checkpoint-base-dir=/data/openpi_checkpoints \
  --exp-name=robocasa365_v21_action_only_v1 \
  --overwrite
~~~

当前默认 global batch size 为 64。若显存不足，先降低 batch size；如果 GPU 显存明显低于 80GB，当前全参数配置仍可能 OOM。

## 6. 单机四卡训练

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run --active scripts/train.py pi05_robocasa365_v21_action_only \
  --data.repo-id="$ROBOCASA_REPO" \
  --checkpoint-base-dir=/data/openpi_checkpoints \
  --exp-name=robocasa365_v21_action_only_v1 \
  --fsdp-devices=4 \
  --overwrite
~~~

## 7. 恢复训练

恢复时不能同时传入 --overwrite：

~~~bash
uv run --active scripts/train.py pi05_robocasa365_v21_action_only \
  --data.repo-id="$ROBOCASA_REPO" \
  --checkpoint-base-dir=/data/openpi_checkpoints \
  --exp-name=robocasa365_v21_action_only_v1 \
  --resume
~~~

## 8. 后续子任务阶段

待获取含 language_persistent/style=subtask 的标注数据后，使用本阶段输出 checkpoint 作为初始化，再切换到：

~~~text
pi05_robocasa365_subtask
~~~

形成两阶段流程：

~~~text
pi05_base
    ↓
RoboCasa365 v2.1 action-only 微调
    ↓
带子任务标注数据的 subtask + action 联合微调
~~~
