# Claude Code × OpenYuanrong 核心代码走读

本文只走读相对 `main` 新增或修改的 **Python / YAML 核心实现**；不包含 `.sh`
启动脚本和 `tests/`。

建议按以下顺序阅读：

```text
1. task_config_claude_code_openyuanrong.yaml  # 要跑什么、在何处跑
2. claude_code/agent.py                       # 如何在沙箱中启动 Claude Code
3. runtime_env.py                             # 如何保证 Ray worker 导入正确包
4. parallel_infer_verl.py                     # 如何把环境放到 Ray job，并拉起引擎
```

## 1. 圆融任务配置：基础镜像、隧道和 Claude Code

文件：`examples/quickstart/training/task_config_claude_code_openyuanrong.yaml`

### 1.1 SWE-Bench 条目

```yaml
- name: swe_bench
  sandbox:
    provider: openyuanrong
    runtime_timeout: 7200
    image_map:
      - from: "swebench/**"
        to: "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/**:v2"
    sandbox_kwargs:
      proxy_port: 38197
      mounts:
        - target: /opt/claude-code
          image_url: swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest
```

逐项解释：

1. `name: swe_bench`
   - 与 Parquet 行 `extra_info.tools_kwargs.task.name` 匹配。
   - 你的 `swe_bench_verified_53_47.parquet` 行会命中这里。

2. `provider: openyuanrong`
   - `get_task(...).build_sandbox()` 会选择 `OpenyuanrongSandbox`，由
     `akernel_sdk` 创建远程沙箱，而不是本地 Docker。

3. `image_map`
   - 数据集保存的是通用名字，例如：

     ```text
     swebench/sweb.eval.x86_64.astropy_1776_astropy-12907
     ```

   - `SandboxConfig._apply_image_map()` 用 `**` 捕获实例后缀，把它变成圆融可用的
     基础任务镜像：

     ```text
     swr.cn-east-3.myhuaweicloud.com/openyuan/swe-bench-verified/
     sweb.eval.x86_64.astropy_1776_astropy-12907:v2
     ```

   - 这个镜像提供 `/testbed` 内的待修复仓库、依赖与评测环境。

4. `proxy_port: 38197`
   - 不是模型端口。
   - 它是圆融沙箱内的反向隧道监听端口。任务运行时，已有的
     `uni_agent.framework.task_runner._inject_gateway_tunnel()` 会把 gateway session
     URL 改写为：

     ```text
     http://127.0.0.1:38197/sessions/<session-id>/v1
     ```

   - 因此沙箱内 Claude Code 请求模型时只连 loopback；圆融负责将请求转发回同节点的
     gateway，再由 gateway 转给 vLLM。

5. `mounts`
   - 这是**第二类镜像**，与上面的基础 SWE 镜像无关。
   - 目标是把 Claude Code sidecar OCI image 以只读方式挂到
     `/opt/claude-code`，供后面的 `agent.executable` 使用。

### 1.2 Claude Code Agent 配置

```yaml
  agent:
    name: claude_code
    executable: /opt/claude-code/bin/claude
    max_turns: 200
    run_timeout: 4800
    model:
      temperature: 1.0
      top_p: 1.0
      max_total_tokens: 131072
```

- `name: claude_code`：选择注册的 `ClaudeCodeAgent`。
- `executable`：不从 PATH 找，而是要求 sidecar 中明确存在这个文件。
- `max_turns: 200`：Claude Code 最多进行 200 轮“模型回复 → 工具调用 → 工具结果”。
- `run_timeout: 4800`：整个 Claude Code 子进程最长 80 分钟。
- `model`：采样与 token 上限会被 veril rollout 使用；运行时的 `base_url`、`api_key`、
  `model_name` 则由 gateway session 覆盖注入。

### 1.3 SWE-rebench 条目

```yaml
- name: swe_rebench
  sandbox:
    image_map:
      - from: "swerebench/sweb.eval.x86_64.**"
        to: "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/**:latest"
```

这里只与 `swe_rebench` 数据有关。其余 Claude Code、隧道和 sidecar 设置与
`swe_bench` 相同。

删除这条 `image_map` 后，`swerebench/sweb.eval.x86_64.<instance>` 会原样传给
圆融，不再改为 SWR 地址。只有圆融明确支持该原始名字作为内网镜像别名时才应删除。
它不影响当前 `swe_bench_verified_53_47.parquet`，也不解决当前 sidecar mount 问题。

---

## 2. ClaudeCodeAgent：支持 sidecar 中的显式可执行文件

文件：`uni_agent/agents/claude_code/agent.py`

### 2.1 新增配置字段

```python
class ClaudeCodeConfig(AgentConfig):
    name: str = "claude_code"
    executable: str = Field(
        default="claude",
        min_length=1,
        description="Claude Code executable name or explicit sidecar path.",
    )
```

以前实现固定执行 `claude`，只能依赖 PATH，无法指定挂载路径。新增这个字段后有两种
模式：

| YAML 值 | 含义 |
|---|---|
| `claude` | 在 PATH 中查找；不存在时可走 npm/native 安装兜底 |
| `/opt/claude-code/bin/claude` | 直接运行 sidecar 文件；不存在时立即报错 |

### 2.2 启动前探测逻辑

新增/修改后的核心逻辑：

```python
async def _ensure_claude(self, sandbox: Sandbox) -> None:
    cfg: ClaudeCodeConfig = self.config
    probe_command = self._claude_probe_command()
    if (await sandbox.exec_shell(probe_command)).exit_code == 0:
        logger.info("claude_code: using preinstalled executable %s", cfg.executable)
        return

    if "/" in cfg.executable:
        raise RuntimeError(
            f"claude_code: configured executable is not executable: {cfg.executable}"
        )

    has_npm = (await sandbox.exec_shell("command -v npm >/dev/null 2>&1")).exit_code == 0
    install_method = "npm" if has_npm else "native installer"
    install_command = _CLAUDE_NPM_INSTALL_COMMAND if has_npm else _CLAUDE_NATIVE_INSTALL_COMMAND
    result = await sandbox.exec_shell(install_command, timeout=_CLAUDE_INSTALL_TIMEOUT)
    ...
```

对应的 probe：

```python
def _claude_probe_command(self) -> str:
    cfg: ClaudeCodeConfig = self.config
    executable = shlex.quote(cfg.executable)
    if "/" in cfg.executable:
        return f"test -x {executable}"
    return f"command -v {executable} >/dev/null 2>&1"
```

这样设计的原因：

- 对显式 sidecar 路径，sidecar 缺失应立即暴露，而不是偷偷在线安装一个不可复现版本。
- 对普通 `claude`，仍保留原有自动安装能力。
- 通过 `shlex.quote()` 避免 YAML 中路径含特殊字符时形成错误 shell 命令。

当前报错正来自这段保护：

```text
claude_code: configured executable is not executable:
/opt/claude-code/bin/claude
```

它说明 agent 在启动模型请求前就发现 sidecar 没有实际挂载到该位置；这比执行到中途再
报“command not found”更明确。

### 2.3 真正执行 CLI 时的改动

```python
argv = [
    cfg.executable,
    "-p",
    user_prompt,
    "--model",
    model,
    "--permission-mode",
    "bypassPermissions",
]
```

这里将以前固定的 `"claude"` 改为 `cfg.executable`。所以只要 sidecar 确实落在配置的
路径，后续 Claude Code 的 shell、读写文件、测试都会在 `/testbed` 内自动执行。

---

## 3. runtime_env.py：解决 Ray worker 的 vLLM 导入漂移

文件：`uni_agent/runtime_env.py`（新增）

完整新增函数：

```python
def pythonpath_with_resolved_package_roots(
    package_names: Iterable[str],
    *existing_pythonpaths: str | None,
) -> str:
    entries: list[str] = []

    for package_name in package_names:
        module = importlib.import_module(package_name)
        module_file = getattr(module, "__file__", None)
        if module_file:
            entries.append(str(Path(module_file).resolve().parent.parent))

    for pythonpath in existing_pythonpaths:
        if pythonpath:
            entries.extend(entry for entry in pythonpath.split(os.pathsep) if entry)

    return os.pathsep.join(dict.fromkeys(entries))
```

逐步作用：

1. 在 driver 中真正 `import vllm`、`import verl`、`import uni_agent`。
2. 从 `module.__file__` 得到真实源码路径，而不是依赖 editable install 的 hook。
3. `parent.parent` 把：

   ```text
   /mnt/share/.../vllm/vllm/__init__.py
   ```

   变为可用于 `PYTHONPATH` 的：

   ```text
   /mnt/share/.../vllm
   ```

4. 将真实目录前置，并用 `dict.fromkeys()` 保持顺序去重。

原始问题是 driver 可导入正确 vLLM，但 Ray worker 将 `vllm` 解析为没有 `LLM` 的
namespace package：

```text
driver: file=/mnt/share/.../vllm/vllm/__init__.py, has_LLM=True
worker: file=None, path=['/vllm'], has_LLM=False
```

这个 helper 的目标是让二者使用同一个物理源目录。

---

## 4. parallel_infer_verl.py：将正确环境放到第一次 ray.init

文件：`examples/inference/parallel_infer_verl.py`

### 4.1 需要转发的圆融变量

```python
_OPENYUANRONG_RUNTIME_ENV_KEYS = (
    "OPENYUANRONG_SERVER_ADDRESS",
    "OPENYUANRONG_TOKEN",
    "OPENYUANRONG_TUNNEL_SSL_VERIFY",
    "USE_OPENYUANRONG_SDK",
    "SANDBOX_NAME_PREFIX",
)
```

这些变量会被 `OpenyuanrongSandbox._load_sandbox_module()` 使用。其中前两个分别是
圆融控制面地址和 token。代码只复制当前 shell 中**已经设置**的值，不会在仓库内保存
凭据。

### 4.2 构造 Ray job runtime_env

```python
def _build_ray_runtime_env(engine: str) -> dict:
    package_names = (
        ("vllm", "verl", "uni_agent")
        if engine == "vllm"
        else ("sglang", "verl", "uni_agent")
    )
    env_vars = {
        "PYTHONPATH": pythonpath_with_resolved_package_roots(
            package_names,
            os.getenv("PYTHONPATH"),
        )
    }
    for key in _OPENYUANRONG_RUNTIME_ENV_KEYS:
        if value := os.getenv(key):
            env_vars[key] = value
    return {"env_vars": env_vars}
```

输入：当前 driver 的真实包路径和圆融环境变量。

输出：传给 Ray 的：

```python
{
    "env_vars": {
        "PYTHONPATH": "<真实 vllm 根目录>:<真实 verl 根目录>:...",
        "OPENYUANRONG_SERVER_ADDRESS": "...",
        "OPENYUANRONG_TOKEN": "...",
    }
}
```

### 4.3 为什么必须放到最早的 ray.init

最终调用改为：

```python
ray.init(runtime_env=_build_ray_runtime_env(args.engine))
_validate_worker_engine_import(args.engine)
```

而不是把变量晚些时候写入：

```text
config.ray_kwargs.ray_init.runtime_env
```

原因是 standalone 推理执行顺序为：

```text
parse_args
-> ray.init
-> init_config（读取/组装 verl 配置）
-> LLMServerManager.create
-> 创建 CheckpointEngineWorker / vLLM worker
```

若等 `init_config()` 才写 runtime_env，Ray job 已经建立，直接创建的 actor 不会继承该
配置。将它传给第一处 `ray.init()` 后，整个 job 的 vLLM server、gateway、task runner
都会拿到同一环境。

### 4.4 低成本 worker 预检

```python
def _probe_worker_engine_import(engine: str) -> dict:
    package_name = "vllm" if engine == "vllm" else "sglang"
    module = importlib.import_module(package_name)
    if engine == "vllm" and not hasattr(module, "LLM"):
        raise ImportError(
            "Ray worker resolved vllm without its LLM API: "
            f"file={getattr(module, '__file__', None)!r}, "
            f"path={list(getattr(module, '__path__', []))!r}, "
            f"sys.path={sys.path!r}"
        )
    return {
        "package": package_name,
        "file": getattr(module, "__file__", None),
        "path": list(getattr(module, "__path__", [])),
    }


def _validate_worker_engine_import(engine: str) -> None:
    probe = ray.remote(num_cpus=0)(_probe_worker_engine_import)
    details = ray.get(probe.remote(engine))
    logger.info("Ray worker engine import verified: %s", details)
```

这段不申请 NPU（`num_cpus=0`），只在一个 Ray worker 中导入一次引擎包。失败时会输出
`file`、`path`、`sys.path`，在模型加载前就能判定 Python 环境问题。

---

## 5. 当前代码的边界与下一步

已经由真实运行日志验证的代码路径：

```text
Ray runtime_env
-> worker vLLM 导入
-> 单 TP8（NPU 0-7）引擎启动
-> LLMServerManager / gateway
-> OpenYuanrongSandbox.start()
-> ClaudeCodeAgent._ensure_claude()
```

当前停止在：

```text
ClaudeCodeAgent._ensure_claude()
-> test -x /opt/claude-code/bin/claude 失败
```

因此下一处需要修的不是模型、Ray 或 image_map，而是这组配置与圆融 OCI sidecar mount
实际行为之间的差异：

```yaml
mounts:
  - target: /opt/claude-code
    image_url: .../claude-code-tool:latest
agent:
  executable: /opt/claude-code/bin/claude
```

在确认 sidecar 实际挂载内容或圆融当前 mount 约束前，不应随意将 `executable` 改为另一
个猜测路径。

## 6. 只看核心代码差异的命令

```bash
git diff origin/main...HEAD -- \
  examples/quickstart/training/task_config_claude_code_openyuanrong.yaml \
  uni_agent/agents/claude_code/agent.py \
  uni_agent/runtime_env.py \
  examples/inference/parallel_infer_verl.py
```
