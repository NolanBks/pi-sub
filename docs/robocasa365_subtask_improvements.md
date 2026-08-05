# OpenPI π0.5 子任务预测与 RoboCasa365 微调关键改进总结

## 1. 改进目标

本项目基于原生 OpenPI 增加了 π0.5 高层任务子任务拆解能力，并将其接入使用 LeRobot Annotate 标注的 RoboCasa365 数据集。

目标训练流程为：

1. 输入当前机器人观测和高层任务，例如 `close the fridge`。
2. π0.5 自回归预测当前子任务，例如 `reach for the fridge handle`。
3. 动作专家以高层任务和子任务为条件，通过 flow matching 预测动作序列。
4. 使用 `pi05_base` checkpoint 初始化模型，在带子任务标注的 RoboCasa365 数据上进行联合微调。

整体过程可以表示为：

```text
RoboCasa observations + high-level task
                    │
                    ▼
       π0.5 language backbone
                    │
                    ├── subtask cross-entropy loss
                    │
                    ▼
       predicted/current subtask
                    │
                    ▼
         flow-matching action expert
                    │
                    ▼
             12-D action chunk
```

## 2. 相对原生 OpenPI 的主要改进

### 2.1 增加 π0.5 子任务监督训练

原生 OpenPI 的 π0.5 主要以任务语言、视觉观测和状态为条件，直接使用 flow matching 预测动作，没有实现论文中高层任务到子任务的显式生成过程。

修改后的代码增加了：

- 子任务训练 tokenizer。
- 子任务 token 的自回归 attention mask。
- 只对子任务 token 计算交叉熵的 loss mask。
- 子任务语言建模损失。
- 高层任务、机器人状态、真实子任务和动作提示的联合输入格式。

训练文本结构近似为：

```text
Task: close the fridge,
State: <discretized robot state>;
Subtask: reach for the fridge handle
Action:
```

其中：

- `Task` 是 episode 的高层任务。
- `State` 是 π0.5 离散化后的机器人状态。
- `Subtask` 是当前帧对应的监督子任务。
- `Action` 是动作生成阶段的条件提示。

主要实现文件：

- `src/openpi/models/tokenizer.py`
- `src/openpi/transforms.py`
- `src/openpi/models/pi0.py`
- `src/openpi/models/pi0_config.py`

### 2.2 增加两阶段推理

修改后的模型支持两阶段推理：

1. 根据图像、高层任务和机器人状态自回归生成子任务 token。
2. 将生成的子任务和 `Action:` 提示重新拼接到语言上下文中。
3. 动作专家在生成子任务的条件下执行 flow-matching 动作采样。

推理输出不再只有动作，还可以包含：

```python
{
    "actions": ...,
    "subtask_tokens": ...,
    "subtask_token_mask": ...,
    "subtask": "reach for the fridge handle",
}
```

JAX 和 PyTorch policy 都已经支持识别返回字典的两阶段推理接口。

主要实现文件：

- `src/openpi/models/pi0.py`
- `src/openpi/models_pytorch/pi0_pytorch.py`
- `src/openpi/policies/policy.py`
- `src/openpi/transforms.py`

### 2.3 接入 LeRobot `language_persistent`

RoboCasa365 的子任务标注使用 LeRobot Annotate 产生的：

- `language_persistent`
- `language_events`

当前训练只使用 `language_persistent` 中满足以下条件的行：

```python
row["style"] == "subtask"
```

对于当前帧时间戳 `t`，代码选择：

```text
timestamp <= t 的最后一条 subtask persistent row
```

作为当前有效子任务。

例如：

```python
language_persistent = [
    {
        "role": "assistant",
        "content": "reach for the fridge handle",
        "style": "subtask",
        "timestamp": 0.0,
    },
    {
        "role": "assistant",
        "content": "pull the fridge door",
        "style": "subtask",
        "timestamp": 1.5,
    },
]
```

当当前帧时间为 `1.0` 秒时，有效子任务是：

```text
reach for the fridge handle
```

当当前帧时间为 `1.6` 秒时，有效子任务是：

```text
pull the fridge door
```

`language_events` 表示只在特定帧发生的瞬时事件，例如 VQA、语音或 interjection。它目前不参与子任务/action 联合训练，后续可用于事件条件策略或辅助语言任务。

主要实现位于：

```text
src/openpi/transforms.py
└── ExtractActiveSubtask
```

### 2.4 增加子任务边界感知的动作 loss mask

一个 action chunk 可能跨越两个子任务。例如：

```text
当前时刻：1.0 s
action horizon：50
fps：20
下一个子任务开始：1.5 s
```

此时长度为 50 的 action chunk 会覆盖约 2.5 秒，后半部分动作已经属于下一个子任务。如果仍然全部使用当前子任务作为条件，会产生错误监督。

为此增加了：

```python
action_loss_mask
```

规则为：

```text
action_timestamp < next_subtask_timestamp  -> 有效
action_timestamp >= next_subtask_timestamp -> 屏蔽
```

mask 会随 observation 进入 JAX/PyTorch 模型，只作用于 flow-matching action loss。有效动作的 loss 会重新缩放，避免靠近子任务边界的样本因为有效 horizon 较短而整体权重过低。

涉及文件：

- `src/openpi/transforms.py`
- `src/openpi/models/model.py`
- `src/openpi/models/pi0.py`
- `src/openpi/models/pi0_config.py`
- `src/openpi/models_pytorch/pi0_pytorch.py`
- `src/openpi/models_pytorch/preprocessing_pytorch.py`

### 2.5 修正联合损失权重

联合训练目标为：

```text
L = λflow × Lflow + λsubtask × Lsubtask
```

当前 RoboCasa365 配置采用：

```text
λflow = 10
λsubtask = 1
```

即：

```text
L = 10 × flow_matching_loss + subtask_cross_entropy_loss
```

该实现避免了把子任务交叉熵错误放大 10 倍。

新增配置字段：

```python
flow_loss_weight
subtask_loss_weight
```

默认值均为 `1.0`，因此没有启用子任务预测的原有 OpenPI 配置保持兼容。

### 2.6 修复训练和推理子任务注入冲突

LIBERO 等没有真实子任务标注的数据可以使用：

```text
subtask = task prompt
```

作为兼容性占位监督。

原实现可能在推理时也注入 identity subtask，导致 tokenizer 错误进入训练分支，从而缺少两阶段推理所需的 action suffix。

现在 identity subtask 只在样本包含 `actions` 时注入，即只用于训练：

```python
if "actions" not in data:
    # inference: do not inject a fake subtask
```

真实 RoboCasa365 子任务标注始终优先于 identity fallback。

## 3. RoboCasa365 数据适配

### 3.1 数据字段映射

RoboCasa365 数据映射如下：

| RoboCasa365 字段 | OpenPI 字段 | 说明 |
|---|---|---|
| `observation.images.robot0_agentview_left` | `base_0_rgb` | 左侧第三人称相机 |
| `observation.images.robot0_eye_in_hand` | `left_wrist_0_rgb` | 末端相机 |
| `observation.images.robot0_agentview_right` | `right_wrist_0_rgb` | 第二个第三人称相机占用剩余视觉槽 |
| `observation.state` | `state` | 16 维机器人状态 |
| `action` | `actions` | 12 维动作 |
| LeRobot task metadata | `prompt` | 高层任务 |
| `language_persistent` | `subtask` | 当前有效子任务 |

模型内部 action dimension 保持 `32`，RoboCasa365 的 12 维动作会补零到 32 维；推理输出时再截取前 12 维。

### 3.2 RoboCasa policy transform

新增：

```text
src/openpi/policies/robocasa_policy.py
```

其中包括：

- `RoboCasaInputs`
- `RoboCasaOutputs`
- `make_robocasa_example`

`RoboCasaOutputs` 会保留生成的子任务文本和 token，不会只返回动作。

## 4. 训练配置

新增训练配置：

```text
pi05_robocasa365_subtask
```

核心参数：

```python
Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=50,
    train_subtask_prediction=True,
    sample_subtask_prediction=True,
    flow_loss_weight=10.0,
    subtask_loss_weight=1.0,
    max_subtask_len=48,
)
```

初始化 checkpoint：

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

数据配置：

```text
LeRobotRoboCasa365DataConfig
```

默认 RoboCasa 帧率：

```text
20 FPS
```

如果实际数据集帧率不是 20，必须同步修改：

```python
LeRobotRoboCasa365DataConfig(fps=<实际帧率>)
```

否则子任务边界对应的 action loss mask 会产生时间偏差。

## 5. 训练数据流

完整数据处理顺序如下：

```text
LeRobotDataset
    │
    ├── 根据 task_index 生成高层 prompt
    │
    ├── 从 language_persistent 提取当前 subtask
    │
    ├── 计算跨子任务边界的 action_loss_mask
    │
    ├── 重映射 RoboCasa365 字段
    │
    ├── 映射三路图像、状态和动作
    │
    ├── state/action quantile normalization
    │
    ├── π0.5 子任务训练 tokenizer
    │
    └── state/action padding 到 action_dim=32
            │
            ▼
    Observation + action targets
            │
            ├── subtask CE loss
            └── masked flow-matching loss
```

## 6. 使用方式

### 6.1 计算归一化统计量

`scripts/compute_norm_stats.py` 已增加 `repo_id` 参数覆盖能力：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name=pi05_robocasa365_subtask \
  --repo-id=YOUR_ORG/YOUR_ROBOCASA365_DATASET
```

### 6.2 短程 smoke test

正式训练前建议先运行少量 step：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run scripts/train.py pi05_robocasa365_subtask \
  --data.repo-id=YOUR_ORG/YOUR_ROBOCASA365_DATASET \
  --exp-name=robocasa365_smoke_test \
  --num-train-steps=2 \
  --batch-size=2 \
  --no-wandb-enabled \
  --overwrite
```

### 6.3 正式训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run scripts/train.py pi05_robocasa365_subtask \
  --data.repo-id=YOUR_ORG/YOUR_ROBOCASA365_DATASET \
  --exp-name=robocasa365_subtask_v1 \
  --overwrite
```

恢复训练：

```bash
uv run scripts/train.py pi05_robocasa365_subtask \
  --data.repo-id=YOUR_ORG/YOUR_ROBOCASA365_DATASET \
  --exp-name=robocasa365_subtask_v1 \
  --resume
```

恢复时不要同时使用 `--resume` 和 `--overwrite`。

## 7. 数据标注要求

为了保证监督有效，数据应满足：

1. 每个 episode 必须有高层 task。
2. `language_persistent` 中至少包含一条 `style="subtask"` 的有效行。
3. 第一条子任务应从 episode 第 0 帧或时间戳 `0.0` 开始生效。
4. `content` 不能为空。
5. 子任务时间戳应单调递增。
6. 数据集 FPS 必须与训练配置一致。
7. 子任务描述应使用一致的粒度和句式。

推荐子任务描述为短动作短语，例如：

```text
reach for the fridge handle
grasp the fridge handle
pull the fridge door
move the gripper away
```

不建议混入完整高层任务解释、推理过程或多步计划。

## 8. 验证状态

目前已经完成：

- Ruff format 检查。
- Ruff lint 检查。
- Python AST/compileall 检查。
- `git diff --check`。
- float32 时间戳边界 smoke test。
- RoboCasa365 输入输出映射 smoke test。
- `language_persistent` list-of-dicts 测试。
- `language_persistent` dict-of-columns 测试。
- 子任务起始覆盖检查。
- 推理阶段 identity subtask 不注入测试。
- RoboCasa 输出保留生成子任务测试。

新增或扩展的测试文件：

- `src/openpi/transforms_test.py`
- `src/openpi/policies/robocasa_policy_test.py`

完整 GPU 训练和完整 pytest 尚需在安装了项目依赖的 Linux/NVIDIA 环境中验证。

## 9. 当前仍需关注的问题

### 9.1 LeRobot 新旧版本兼容

当前项目仍固定使用原生 OpenPI 的旧 LeRobot revision：

```toml
lerobot = {
    git = "https://github.com/huggingface/lerobot",
    rev = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
}
```

代码已经兼容新旧 LeRobot Python import 路径，但旧 LeRobot reader 可能无法直接读取带 `language_persistent/language_events` 的新 v3.x 数据集。

不能简单在当前环境中直接升级 LeRobot，因为新 LeRobot 版本通常同时要求：

- Python 3.12 或更高。
- NumPy 2.x。
- 更新的 Hugging Face 依赖。

而当前 OpenPI 环境使用：

- Python 3.11。
- NumPy `<2.0`。
- Transformers `4.53.2`。

因此，正式训练前必须先完成单样本数据读取检查。如果 reader 无法读取新数据集，需要单独设计：

1. LeRobot v3 数据兼容环境；
2. 数据预转换流程；或
3. OpenPI 与新 LeRobot 的受控依赖迁移。

### 9.2 `language_events` 尚未用于训练

当前只使用 `language_persistent/style=subtask`。

后续可考虑将 `language_events` 用于：

- VQA 辅助训练。
- 瞬时语言指令。
- 失败恢复提示。
- 人机交互事件。
- 多任务 language head。

但这些功能不应直接混入当前 subtask CE loss，需要独立定义输入格式、目标 mask 和 loss 权重。

### 9.3 需要真实数据端到端验证

目前本地没有实际 RoboCasa365 Annotate 数据样本，因此以下内容仍需要在服务器上确认：

- 实际 `language_persistent` Python 返回结构。
- 数据集真实 FPS。
- `task_index` 与高层任务映射。
- 图像存储格式和颜色通道。
- episode 末尾 action chunk 的处理方式。
- 子任务时间戳是否覆盖整个 episode。
- normalization statistics 是否存在异常值。

### 9.4 显存需求

当前配置是完整 π0.5 微调，不是 LoRA 微调。单 GPU 通常需要约 80 GB 级别显存。

如果服务器显存不足，后续需要增加专用的子任务 LoRA 配置，包括：

- PaliGemma LoRA。
- action expert LoRA 或部分解冻。
- 对应 freeze filter。
- 更小 batch size。
- gradient accumulation 或多 GPU FSDP。

## 10. 关键文件清单

| 文件 | 关键改进 |
|---|---|
| `src/openpi/models/tokenizer.py` | 子任务训练/推理 tokenizer |
| `src/openpi/transforms.py` | persistent 子任务提取、边界 mask、tokenization、detokenization |
| `src/openpi/models/model.py` | Observation 增加子任务推理字段和 action loss mask |
| `src/openpi/models/pi0.py` | JAX 子任务 CE、两阶段推理、masked flow loss |
| `src/openpi/models/pi0_config.py` | 子任务配置和 loss 权重配置 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | PyTorch 子任务训练和两阶段推理 |
| `src/openpi/models_pytorch/preprocessing_pytorch.py` | 保留子任务和 action mask 字段 |
| `src/openpi/policies/policy.py` | policy 支持模型返回动作与子任务字典 |
| `src/openpi/policies/robocasa_policy.py` | RoboCasa365 输入输出适配 |
| `src/openpi/training/config.py` | RoboCasa365 data config 和 train config |
| `src/openpi/training/data_loader.py` | LeRobot 新旧 import 路径兼容 |
| `scripts/compute_norm_stats.py` | 支持通过 CLI 覆盖数据集 repo ID |
| `src/openpi/transforms_test.py` | persistent 语言和边界测试 |
| `src/openpi/policies/robocasa_policy_test.py` | RoboCasa365 映射与输出测试 |

## 11. 总结

当前项目已经完成从：

```text
高层 task
```

到：

```text
当前 subtask
```

再到：

```text
subtask-conditioned action chunk
```

的核心代码闭环，并完成 RoboCasa365 三路视觉、状态、动作以及 LeRobot persistent 子任务标注的训练适配。

在正式训练之前，剩余的首要任务不是继续修改模型结构，而是验证服务器上的实际数据能否通过当前 LeRobot reader 正确返回 `language_persistent`。完成这一检查后，即可计算 normalization statistics、执行短程 smoke test，并从 `pi05_base` 开始正式微调。
