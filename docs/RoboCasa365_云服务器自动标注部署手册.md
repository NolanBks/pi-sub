# RoboCasa365 云服务器自动标注部署手册（A800 / 多卡 / 本地 VLM checkpoint）

> 目标：在一台配备 A800（1、2 或 4 张卡）的 Linux 云服务器上，将 RoboCasa365 数据转为/保持为 LeRobot v3 数据集，并通过 **LeRobot `lerobot-annotate` + 本地 vLLM 服务**自动写入可训练的长程语言标注。
>
> 这份手册选择的是目前较省心的一体化方案：LeRobot 负责视频读取、抽帧、长程 task/plan/subtask 生成、时间分段、schema 校验和写回；vLLM 仅负责承载你已有的本地 VLM checkpoint。不要再另起一个旧版的 LeRobot Web 标注 UI。

---

## 1. 最终会得到什么

每个 episode 会在数据集中新增或更新如下语言字段：

- `language_persistent`: `task`、`task_aug`、`plan`、`subtask`、`memory` 等跨片段、可供长程策略使用的文字。
- `language_events`: `interjection`、`speech`、`vqa` 等事件型文字（可选，成本较高）。
- 标注质量控制：LeRobot 会先写入 `.annotate_staging/`，完成 validator 检查后再合并回数据集；失败的 episode 可按清单重跑。

建议分两阶段跑：

1. **主标注（必跑）**：task + plan + subtask + memory。它最适合 RoboCasa365 的长程任务分解。
2. **事件补充（可选）**：interjection / VQA。仅在你确实要训练问答、纠错或语言介入能力时打开。

---

## 2. 设计选择与资源规划

### 推荐拓扑

```text
RoboCasa365 LeRobot dataset (本地 SSD / NAS 挂载)
             │
             ▼
       lerobot-annotate
  抽帧 → 提示词 → 分段 → 校验 → Parquet 写回
             │ OpenAI-compatible HTTP
             ▼
       vLLM（本地 checkpoint）
             │
             ▼
           A800 GPU
```

### A800 配置建议

| 资源 | 建议模型规模 | 启动方式 | 适用场景 |
|---|---:|---|---|
| 1 × A800 80G | 7B/8B VLM | TP=1 | 冒烟测试、低成本全量 |
| 2 × A800 80G | 20B–32B VLM | TP=2 | **质量/成本最推荐** |
| 4 × A800 80G | 20B–32B VLM | 两个 TP=2 vLLM 副本 | 高吞吐全量标注 |

TP 是 tensor parallel。四卡时不要让两个 `lerobot-annotate` 进程同时写同一个数据集目录；使用**一个** annotator，令它管理两个 vLLM server 副本即可。

### 磁盘与时间预估

- 数据迁移会原地换根目录；先复制一份数据，预留至少约 **2× 原始数据体积**的空间。
- 建议数据和 staging 都放本地 NVMe，不要直接在高延迟对象存储上随机读取视频。
- 先跑 8–16 个 episode 校验输出与相机视角，再跑全量。
- 打开 VQA / interjection 会显著增加 VLM 请求数；主标注稳定后再补跑。

---

## 3. 约定路径和变量

以下命令假设你已通过 SSH 登录 Ubuntu 22.04/24.04 云服务器，并把变量替换为实际路径。所有后续 shell 都沿用这些变量。

```bash
export RC_WORK=/data/robocasa_annotation
export RC_DATA_SRC=/data/robocasa365_raw              # 只读原始数据
export RC_DATA_OUT=/data/robocasa365_lerobot_v3       # 标注数据副本
export RC_CKPT=/models/your-local-vlm-checkpoint       # 你的本地 VLM 模型目录
export RC_IMAGE=robocasa-annotator:0.6.1
export RC_CAMERA=observation.images.front              # 后续用实际 key 替换

mkdir -p "$RC_WORK" "$RC_DATA_OUT"
```

checkpoint 必须至少包含模型权重、`config.json`、tokenizer 和图像 processor/preprocessor 配置；并且必须是 **vLLM 可加载、支持多图输入的 VLM**。如果模型自身要求自定义代码，后面的 vLLM 命令可加入 `--trust-remote-code`，但只应对可信 checkpoint 启用。

---

## 4. 服务器基础环境

### 4.1 检查 GPU 与 Docker

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

若最后一条无法看到 GPU，先安装/修复 NVIDIA Container Toolkit，再继续：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4.2 构建固定版本镜像

在 `$RC_WORK` 创建 `Dockerfile`：

```dockerfile
FROM vllm/vllm-openai:latest

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --no-deps \
      "lerobot @ git+https://github.com/huggingface/lerobot.git@v0.6.1" && \
    python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
      "datasets>=4.7.0,<5.0.0" "pyarrow>=21.0.0,<30.0.0" \
      "av>=15.0.0,<16.0.0" "draccus==0.10.0" \
      "pandas>=2.0.0,<3.0.0" jsonlines gymnasium torchcodec \
      mergedeep pyyaml-include toml typing-inspect openai

ENV VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
ENV VLLM_VIDEO_BACKEND=pyav
WORKDIR /workspace
```

构建并确认命令存在：

```bash
cd "$RC_WORK"
docker build -t "$RC_IMAGE" .
docker run --rm --gpus all "$RC_IMAGE" lerobot-annotate --help
```

如果你已有可用的内部镜像，只要它同时含有 `vllm`、`lerobot-annotate`、FFmpeg/PyAV 依赖，也可跳过 Dockerfile；但建议把镜像 tag 和 pip 版本记录在实验日志中。

---

## 5. 准备并检查数据

### 5.1 永远从副本开始

```bash
cp -a "$RC_DATA_SRC/." "$RC_DATA_OUT/"
du -sh "$RC_DATA_SRC" "$RC_DATA_OUT"
```

不要把 `RC_DATA_OUT` 指向唯一的原始 RoboCasa365 数据。LeRobot 的转换和标注都会写入该目录。

### 5.2 查看当前 dataset version 与相机 key

```bash
jq ".codebase_version, .total_episodes, .total_frames" "$RC_DATA_OUT/meta/info.json"
jq -r ".features | to_entries[] | select(.value.dtype == \"video\") | .key" "$RC_DATA_OUT/meta/info.json"
```

将输出的、最能看清桌面主任务的固定机位写入 `RC_CAMERA`。不要盲用默认第一个视频流：RoboCasa365 中第一个可能是 wrist camera，长程子任务判断常会变差。

```bash
export RC_CAMERA=observation.images.front  # 例子，必须按上一条输出修改
```

### 5.3 若数据仍是 LeRobot v2.1：转换到 v3

`lerobot-annotate` 面向 v3.1 数据集。若 `meta/info.json` 中版本是 `v2.1`，先在**副本**上转换。转换器会暂存旧目录为 `<root>_old`，所以双倍空间是必要条件。

```bash
docker run --rm --gpus all --ipc=host \
  -v "$RC_DATA_OUT:/dataset" \
  "$RC_IMAGE" bash -lc "\
    LEROBOT_DIR=\$(python -c \"import pathlib, lerobot; print(pathlib.Path(lerobot.__file__).parent)\") && \
    python \$LEROBOT_DIR/scripts/convert_dataset_v21_to_v30.py \
      --repo-id=local/robocasa365 \
      --root=/dataset \
      --push-to-hub=false"
```

转换完成后重新执行 5.2。如果你的数据不是 LeRobot 格式，而是原始 RoboCasa HDF5/MJCF，请先使用你的 RoboCasa 导出脚本生成 LeRobot v3 数据；本手册不建议直接对 HDF5 目录跑 `lerobot-annotate`。

---

## 6. 先验证本地 VLM checkpoint

先单独启动 vLLM，确认模型、图像处理器和显存都没有问题。以下示例使用一张卡和端口 8000：

```bash
docker run --rm --gpus '"device=0"' --ipc=host -p 8000:8000 \
  -v "$RC_CKPT:/model:ro" \
  "$RC_IMAGE" \
  vllm serve /model \
    --served-model-name robocasa-local-vlm \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt "{\"image\": 64}"
```

若模型需要自定义 remote code，在命令末尾增加 `--trust-remote-code`。对于 Qwen3.6 等带 thinking 的模型，可额外在 LeRobot 命令中添加：

```bash
--vlm.chat_template_kwargs='{"enable_thinking": false}'
```

在另一终端验证服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

看到 `robocasa-local-vlm` 即通过。按 `Ctrl+C` 停掉手工服务；下一节由 `lerobot-annotate` 自动拉起 vLLM。

---

## 7. 第一次冒烟标注：8 个 episode、单卡

先运行核心语言标注。这里通过 `--vlm.serve_command` 让 LeRobot 自行启停 vLLM，避免手工维护后台进程。

```bash
export RC_EPISODES="[0,1,2,3,4,5,6,7]"

docker run --rm --gpus '"device=0"' --ipc=host \
  -v "$RC_DATA_OUT:/dataset" \
  -v "$RC_CKPT:/model:ro" \
  -v "$RC_WORK:/workspace" \
  "$RC_IMAGE" \
  lerobot-annotate \
    --root=/dataset \
    --only_episodes="$RC_EPISODES" \
    --modalities='[plan]' \
    --vlm.backend=openai \
    --vlm.model_id=robocasa-local-vlm \
    --vlm.camera_key="$RC_CAMERA" \
    --vlm.num_gpus=1 \
    --vlm.parallel_servers=1 \
    --vlm.client_concurrency=2 \
    --executor.episode_parallelism=1 \
    --vlm.serve_command='vllm serve /model --served-model-name robocasa-local-vlm --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 --limit-mm-per-prompt "{\"image\": 64}" --port {port}'
```

`modalities=[plan]` 是推荐的核心模式：它会产生任务描述、长程计划、subtask 分解和相关 persistent annotations。模型若 OOM，先按以下顺序调低：`client_concurrency=1`、`max-model-len=4096`、`limit-mm-per-prompt` 中 image 数量、模型规模。

---

## 8. 全量运行

### 8.1 2 × A800：推荐的质量/成本配置

确认冒烟结果合理后，去掉 `--only_episodes` 即可跑全量。TP=2 表示同一个 VLM 跨两张 GPU：

```bash
docker run --rm --gpus all --ipc=host \
  -v "$RC_DATA_OUT:/dataset" \
  -v "$RC_CKPT:/model:ro" \
  -v "$RC_WORK:/workspace" \
  "$RC_IMAGE" \
  lerobot-annotate \
    --root=/dataset \
    --modalities='[plan]' \
    --vlm.backend=openai \
    --vlm.model_id=robocasa-local-vlm \
    --vlm.camera_key="$RC_CAMERA" \
    --vlm.num_gpus=2 \
    --vlm.parallel_servers=1 \
    --vlm.client_concurrency=4 \
    --executor.episode_parallelism=2 \
    --vlm.serve_command='vllm serve /model --served-model-name robocasa-local-vlm --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.88 --limit-mm-per-prompt "{\"image\": 64}" --port {port}' \
  2>&1 | tee "$RC_WORK/annotate-plan-$(date +%F-%H%M).log"
```

### 8.2 4 × A800：两个 TP=2 副本提高吞吐

这仍是单个 annotator writer；它自动拉起两个 vLLM server。每个服务使用两张卡：

```bash
docker run --rm --gpus all --ipc=host \
  -v "$RC_DATA_OUT:/dataset" \
  -v "$RC_CKPT:/model:ro" \
  -v "$RC_WORK:/workspace" \
  "$RC_IMAGE" \
  lerobot-annotate \
    --root=/dataset \
    --modalities='[plan]' \
    --vlm.backend=openai \
    --vlm.model_id=robocasa-local-vlm \
    --vlm.camera_key="$RC_CAMERA" \
    --vlm.num_gpus=2 \
    --vlm.parallel_servers=2 \
    --vlm.client_concurrency=8 \
    --executor.episode_parallelism=4 \
    --vlm.serve_command='vllm serve /model --served-model-name robocasa-local-vlm --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.88 --limit-mm-per-prompt "{\"image\": 64}" --port {port}' \
  2>&1 | tee "$RC_WORK/annotate-plan-4gpu-$(date +%F-%H%M).log"
```

如果你的模型只能 TP=1，可改成 `num_gpus=1`、`parallel_servers=4`；但 7B 模型速度好，不代表长程细粒度分段质量一定好。

### 8.3 断点续跑

不要删除 `.annotate_staging/`，它记录正在处理的临时结果。若中断，优先使用失败 episode 的列表加 `--only_episodes` 重跑；已写入并通过校验的 episode 不应重复覆盖。运行前后都保留日志。

---

## 9. 可选：补充 interjection / VQA

核心 plan 标注确认没问题后，再针对相同数据跑事件型标注。因为 VQA 和 interjection 的调用量更大，建议先抽 100 个 episode 验证。

```bash
export RC_EPISODES="[0,1,2,3,4,5,6,7]"

docker run --rm --gpus all --ipc=host \
  -v "$RC_DATA_OUT:/dataset" \
  -v "$RC_CKPT:/model:ro" \
  "$RC_IMAGE" \
  lerobot-annotate \
    --root=/dataset \
    --only_episodes="$RC_EPISODES" \
    --modalities='[interjection,vqa]' \
    --vlm.backend=openai \
    --vlm.model_id=robocasa-local-vlm \
    --vlm.camera_key="$RC_CAMERA" \
    --vlm.num_gpus=2 \
    --vlm.parallel_servers=1 \
    --vlm.client_concurrency=4 \
    --executor.episode_parallelism=2 \
    --vlm.serve_command='vllm serve /model --served-model-name robocasa-local-vlm --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.88 --limit-mm-per-prompt "{\"image\": 64}" --port {port}'
```

若你的下游仅需要“任务 → 子任务片段”监督数据，不建议默认跑这一步。

---

## 10. 检查标注结果

### 10.1 检查数据集字段

```bash
docker run --rm -v "$RC_DATA_OUT:/dataset:ro" "$RC_IMAGE" bash -lc '
python - << "PY"
import json
info = json.load(open("/dataset/meta/info.json"))
for k in info.get("features", {}):
    if "language" in k or k in {"task", "task_index"}:
        print(k)
PY
'
```

### 10.2 抽样读取前几条 Parquet

```bash
docker run --rm -v "$RC_DATA_OUT:/dataset:ro" "$RC_IMAGE" bash -lc '
python - << "PY"
import glob
import pandas as pd
files = sorted(glob.glob("/dataset/data/**/*.parquet", recursive=True))
print("parquet files:", len(files))
df = pd.read_parquet(files[0])
cols = [c for c in df.columns if "language" in c or "task" in c]
print(df[cols].head(10).to_string())
PY
'
```

抽检时重点看：

- task 是否描述的是完整 RoboCasa 任务，而不是单帧里某个物体；
- plan 是否按可执行顺序分解；
- subtask 的时间边界是否和手部/物体接触、开关、放置等变化一致；
- 失败、空轨迹、视野遮挡 episode 是否被正确标记或剔除。

建议由人工随机审核至少 100 个 episode，并保存：episode id、是否通过、错误类型、修改建议。若同类错误超过约 10%，先调整相机选择或更换/微调 VLM，再继续全量。

---

## 11. 常见问题

| 现象 | 优先排查 / 处理 |
|---|---|
| `No module named lerobot` 或没有 `lerobot-annotate` | 重新 `docker build`，并执行 `docker run ... lerobot-annotate --help`。 |
| vLLM 不能加载 checkpoint | 先按第 6 节单独运行 vLLM；检查模型架构是否被 vLLM 支持、模型文件是否完整、是否确实需要 `--trust-remote-code`。 |
| CUDA OOM | 降 `client_concurrency`，再降 `max-model-len`/图像数，最后换更小模型或增加 TP 卡数。 |
| 模型服务启动但 annotate 连接失败 | 保持 `--served-model-name` 与 `--vlm.model_id` 完全相同；`serve_command` 必须包含 `--port {port}`。 |
| 生成的 subtask 很泛或不连续 | 换到外部固定机位；确认视频 key；用较强 VLM；先检查采样的 8 episodes，而非直接加并发。 |
| 四卡速度没有提升 | 检查是否设置了 `parallel_servers=2`、`num_gpus=2` 和 vLLM `--tensor-parallel-size 2`；执行时用 `nvidia-smi` 看 4 卡是否都有负载。 |
| 数据集被意外改坏 | 原始数据应始终只读保存。工作副本可从备份恢复；因此不要跳过第 5.1 节。 |

---

## 12. 上线前清单

- [ ] 原始 RoboCasa365 数据已保留，`RC_DATA_OUT` 是副本。
- [ ] `meta/info.json` 表明数据已是 LeRobot v3，并确认了实际 video camera key。
- [ ] 本地 checkpoint 已能用 `vllm serve` 运行并通过 `/v1/models` 测试。
- [ ] 8 个 episode 的 `plan` 标注已经人工检查。
- [ ] 已按 GPU 数选择 TP/并行服务数；日志输出到 `$RC_WORK`。
- [ ] 全量结束后抽检至少 100 个 episode，并记录质量结论。

---

## 13. 官方参考

- [LeRobot Annotation Pipeline](https://huggingface.co/docs/lerobot/main/annotation_pipeline)
- [LeRobot Dataset v3 格式](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [LeRobot GitHub 仓库](https://github.com/huggingface/lerobot)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

---

## 14. 需要按你的 checkpoint 微调的唯一部分

本手册刻意把模型型号抽象成 `robocasa-local-vlm`。部署前只需根据模型卡确认以下四点：

1. 该 checkpoint 在 vLLM 中的正确加载路径和是否需要 `--trust-remote-code`；
2. 最大上下文长度（对应 `--max-model-len`）；
3. 多图限制（对应 `--limit-mm-per-prompt`）；
4. 如果是 Qwen 系列，是否需要关闭 thinking 的 `chat_template_kwargs`。

其他数据管线、schema、分段和写回流程均不需要因模型而重写。
