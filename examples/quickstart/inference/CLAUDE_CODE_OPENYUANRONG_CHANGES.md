# Claude Code × OpenYuanrong 改动说明

本文档用于对比 `main` 与分支 `codex/claude-code-openyuanrong`。目标是在
SWE-Bench / SWE-rebench 任务中，使用本地 Qwen 模型作为策略模型，在圆融远程
沙箱内运行 Claude Code，并通过 verl 的 gateway 记录轨迹和回传 reward。

> 当前结论：**8 卡 Qwen/vLLM、Ray worker 导入、gateway 和圆融 Sandbox 创建均已
> 验证通过。** 当前尚未完成的是 Claude Code sidecar 的 OCI mount：沙箱内
> `/opt/claude-code` 不存在，因此 Claude Code 还没有真正启动。

## 1. 端到端流程

```text
Parquet 样本
  -> TaskConfigResolver（合并 YAML + 样本中的任务镜像）
  -> OpenYuanrong Sandbox（SWE 基础镜像）
  -> Claude Code CLI（预期由 sidecar mount 提供）
  -> 127.0.0.1:38197（圆融反向隧道）
  -> uni-agent Gateway session
  -> verl / vLLM / Qwen3.5-35B-A3B
  -> Claude Code 工具调用、修改 /testbed、运行测试
  -> SWE reward 回传 TransferQueue
```

这里的 Claude Code 是**实际 CLI**，但其模型请求会指向本地 Qwen/vLLM，不会访问
Anthropic 的 Claude 服务。

## 2. 新增 / 修改文件一览

| 文件 | 类型 | 主要内容 | 为什么要改 |
|---|---|---|---|
| `examples/quickstart/training/task_config_claude_code_openyuanrong.yaml` | 新增 | 圆融的 `swe_bench` / `swe_rebench` 任务配置 | 将数据集任务接到圆融沙箱与 Claude Code |
| `uni_agent/agents/claude_code/agent.py` | 修改 | `agent.executable` 支持显式 sidecar 路径 | 允许使用挂载的 Claude Code，而非每次在线安装 |
| `examples/inference/parallel_infer_verl.py` | 修改 | Ray job 环境、物理包路径、worker 导入预检 | 修复 driver 与 Ray worker 的 vLLM 导入不一致 |
| `uni_agent/runtime_env.py` | 新增 | 解析真实 Python 包根目录并构造 `PYTHONPATH` | 避免 editable vLLM 在 worker 中变成 `/vllm` namespace package |
| `examples/quickstart/inference/run_single_infer_claude_code_openyuanrong.sh` | 新增 | 你的单样本启动脚本 | 固定数据、模型、日志、8 卡参数并做前置检查 |
| `examples/quickstart/inference/smoke_infer_claude_code_openyuanrong.sh` | 新增 | `ray job submit` 方式的 smoke 示例 | 用于以 Ray Job 提交、携带私有 runtime-env 的场景 |
| `examples/quickstart/training/runtime_env_openyuanrong.example.yaml` | 新增 | 圆融 token/地址的私有配置模板 | 避免把凭据提交到仓库 |
| `examples/quickstart/training/train_qwen3p5_dense.sh` | 修改 | 训练入口向 Ray worker 传圆融环境变量 | 训练 path 下也可创建圆融沙箱 |
| `tests/uni_agent/...` | 新增/修改 | 配置合并、Claude executable、runtime-env 测试 | 防止关键配置回归 |

查看某个文件的完整改动：

```bash
git diff origin/main...HEAD -- <文件路径>
```

查看全部实现改动（忽略测试）：

```bash
git diff origin/main...HEAD -- \
  ':!tests/**'
```

## 3. 圆融任务 YAML

文件：`examples/quickstart/training/task_config_claude_code_openyuanrong.yaml`

它有两个条目，分别由 Parquet 行里的 `task.name` 选择：

```yaml
- name: swe_bench
  sandbox:
    provider: openyuanrong
  agent:
    name: claude_code

- name: swe_rebench
  sandbox:
    provider: openyuanrong
  agent:
    name: claude_code
```

### 3.1 基础任务镜像的 image_map

数据集只保存通用镜像名，避免把某个 provider 的镜像地址写死到数据里。例如：

```text
swe_bench:   swebench/sweb.eval.x86_64.<instance>
swe_rebench: swerebench/sweb.eval.x86_64.<instance>
```

YAML 在真正创建圆融 Sandbox 前做映射：

```yaml
# SWE-Bench Verified
from: "swebench/**"
to: "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/**:v2"

# SWE-rebench
from: "swerebench/sweb.eval.x86_64.**"
to: "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/**:latest"
```

`**` 会保留实例名。比如：

```text
swerebench/sweb.eval.x86_64.astropy_1776_astropy-12907
-> swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/
   astropy_1776_astropy-12907:latest
```

这两条 map **只影响基础 SWE 沙箱镜像**，不影响 Qwen 模型，也不影响 Claude Code
sidecar。

### 3.2 是否能删除 swe_rebench 的 image_map

可以删除，但只建议在圆融平台确认 `swerebench/...` 是其可直接识别的内网镜像别名时
删除。删除后的行为是：

```text
删除前：swerebench/... -> swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/...:latest
删除后：swerebench/... 原样传给 OpenYuanrong Sandbox
```

影响范围：

| 情况 | 删除映射后的结果 |
|---|---|
| 圆融可以直接拉取/解析 `swerebench/...` 内网镜像 | 可以运行，且不依赖 SWR 映射 |
| 圆融不能解析该别名 | Sandbox 基础镜像拉取失败，或拉到错误镜像 |
| 当前 `swe_bench_verified_53_47.parquet` 单样本运行 | **没有影响**，因为它走的是 `swe_bench` 条目 |

因此它不能解决当前 `/opt/claude-code` 缺失的问题；那是另一个 sidecar mount 问题。

### 3.3 Claude Code sidecar 与反向隧道

YAML 目前声明：

```yaml
sandbox_kwargs:
  proxy_port: 38197
  mounts:
    - target: /opt/claude-code
      image_url: swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest

agent:
  executable: /opt/claude-code/bin/claude
```

含义：

- `proxy_port: 38197`：圆融在沙箱内建立到 gateway 的反向隧道；
  `run_task` 会把 gateway session URL 改写为 `http://127.0.0.1:38197/...`。
- `mounts`：预期将 Claude Code OCI image 只读挂到 `/opt/claude-code`。
- `executable`：Claude Code Agent 在 `/testbed` 工作目录中执行该 CLI。

当前已知问题：SDK 的 `Mount(target, image_url, type="bind")` 参数构造正确，但实际
Sandbox 中 `/opt/claude-code` 不存在。必须先确认当前圆融运行时的 OCI mount 行为或
可用的 sidecar 目标路径，不能盲目改 `executable`。

## 4. Claude Code Agent 的改动

文件：`uni_agent/agents/claude_code/agent.py`

新增的配置：

```python
executable: str = Field(default="claude")
```

执行时的行为：

1. 如果是名称（如 `claude`），先执行 `command -v claude`；不存在时允许 npm/native
   installer 安装。
2. 如果是绝对路径（如 `/opt/claude-code/bin/claude`），先执行 `test -x`。
3. 显式路径不存在时立即失败，不偷偷进行网络安装。

这样做的原因是：预装 sidecar 缺失时，如果无提示地在线安装，可能在沙箱中访问外网、
拉取不确定版本，且难以判断真正的问题是 mount 还是安装失败。

实际 CLI 启动方式仍然是：

```text
claude -p <任务描述> --model Qwen3.5-35B-A3B \
  --permission-mode bypassPermissions --max-turns 200
```

它会得到：

```text
ANTHROPIC_BASE_URL=http://127.0.0.1:38197/<session>
ANTHROPIC_MODEL=Qwen3.5-35B-A3B
```

因此 CLI 的工具调用（读文件、编辑、shell、测试）发生在圆融沙箱，而模型回复由 Qwen
产生。

## 5. Ray / vLLM 导入修复

### 问题

driver 进程能导入：

```text
/mnt/share/t00986241/latest_release/vllm/vllm/__init__.py
```

但 Ray worker 曾解析为：

```text
file=None, path=['/vllm']
```

因此 worker 报：

```text
ImportError: cannot import name 'LLM' from 'vllm'
```

### 新增 helper

文件：`uni_agent/runtime_env.py`

`pythonpath_with_resolved_package_roots()` 在 driver 中导入 `vllm`、`verl`、
`uni_agent`，取它们真实 `__file__` 的父目录，并放在 worker `PYTHONPATH` 的最前面。

### 为什么改到第一次 ray.init

最初曾把 `PYTHONPATH` 写入 verl 配置的：

```text
ray_kwargs.ray_init.runtime_env
```

但 standalone 推理脚本读取 YAML 前已经执行了 `ray.init()`，所以此配置不会影响直接
创建的 `CheckpointEngineWorker`。现在改为：

```python
ray.init(runtime_env=_build_ray_runtime_env(args.engine))
```

这样当前 job 创建的 vLLM、gateway、task runner 等 Ray actor 都继承同一套物理包路径和
圆融凭据。

此外，启动后会先跑一个轻量 Ray worker import 预检：

```python
from vllm import LLM
```

避免完整模型初始化后才发现包解析错误。

## 6. 单样本 8 卡启动脚本

文件：`examples/quickstart/inference/run_single_infer_claude_code_openyuanrong.sh`

默认固定为：

```bash
DATA_PATH=/mnt/share/z00876269/datasets/uniagent_0901/swe_bench_verified_53_47.parquet
MODEL_PATH=/mnt/share/z00876269/models/Qwen3.5-35B-A3B
TP=8
N_DEVICES_PER_NODE=8
--limit 1
--concurrency 1
```

原因：35B-A3B 使用一个 TP8 模型副本足以进行单样本 smoke；此前 16 卡配置会创建两个
TP8 副本，其中一组曾被调度到 NPU 8–15，并在 NPU 9/11 出现 Ascend HDC 启动超时。

需要16卡吞吐时，不改文件，直接覆盖：

```bash
N_DEVICES_PER_NODE=16 \
bash examples/quickstart/inference/run_single_infer_claude_code_openyuanrong.sh
```

## 7. 当前验证状态

| 层级 | 状态 | 证据 |
|---|---|---|
| Ray 连接 | 已通过 | driver 能连接 `80.5.25.123:27894` |
| Ray worker vLLM 导入 | 已通过 | worker 可导入 `vllm.LLM` |
| 单副本 TP8 引擎 | 已通过 | `visible_npus=[0,1,2,3,4,5,6,7]`、`LLMServerManager` 启动 |
| 圆融 SDK 初始化 | 已通过 | `ensure_yr_init()` 成功 |
| 圆融 Sandbox 基础创建 | 已通过过一次 | 后续已进入 `ClaudeCodeAgent._ensure_claude` |
| Claude sidecar 网络拉取 | 曾有短暂超时 | 圆融节点访问 SWR 的 443 超时；随后 Sandbox 可创建 |
| Claude sidecar 实际挂载 | 未通过 | `/opt/claude-code` 不存在 |
| Claude Code 多轮工具调用 | 未开始 | 缺少可执行的 `claude` |
| SWE 评分 / result.json | 未开始 | Agent 尚未运行完 |

## 8. 不应误解的事项

- `pynvml`、`requests` 版本兼容、`migratepages`、RL-Insight 等日志是 warning，不是当前
  失败根因。
- `image_map` 是基础任务镜像；`mounts.image_url` 是 Claude Code sidecar，二者是两条
  独立链路。
- 删除 `swe_rebench` 的 image_map 不会修复 Claude sidecar mount，也不会影响当前
  `swe_bench_verified_53_47.parquet` 单样本运行。
