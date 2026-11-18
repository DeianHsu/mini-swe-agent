# mini-swe-agent-demo

一个参考 [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) 手写的 Python 智能体（Agent）学习项目。

模型通过原生函数调用（Function Calling）提交 Bash 命令，在 Docker 容器中读取文件、修改代码和运行测试。程序将执行结果写回消息历史，驱动下一轮操作。

## 功能

- 单一工具 `execute`：解析 `tool_calls` 中的 `command`，交给容器中的 `bash -c` 执行。
- 工具结果通过关联 `tool_call_id` 的 `tool` 消息反馈给模型。
- 模型停止工具调用后结束，默认最多调用模型 250 轮，单条命令执行上限 60 秒。
- 由 `budget.py` 监管每题 50 分钟时限，费用阈值为 3 美元；`cost.py` 负责词元和费用统计。
- 默认关闭思考；预算入口可用 `--thinking-mode max` 启用最高档，模式随轨迹和结果保存。
- 单次命令输出最多 16,000 字符，工具历史正文累计最多 256,000 字符；截断时保留调用对应关系。

## 项目结构

```text
src/
└── swe/
    ├── __init__.py
    ├── mini_agent.py        # 消息历史、工具调度与智能体循环
    ├── environment.py       # Docker Bash 命令执行与输出截断
    ├── use_llm.py           # 模型请求与原生函数调用定义
    ├── budget.py            # 预算监管与运行结果保存
    └── cost.py              # 词元与费用统计
```

`ask_model()` 组织提示与消息历史，`run_agent()` 控制循环并调度工具，`execute()` 执行命令并返回退出码、标准输出和标准错误。每次命令启动独立进程，容器内文件修改会保留，跨次调用的 `cd` 和环境变量设置不会保留。

## 安装与配置

需要 Python 3.14+、uv 和运行中的 Docker Linux 引擎。Windows 可使用 Docker Desktop。

在项目根目录安装依赖：

```powershell
uv sync
```

在根目录创建 `.env`，填写模型服务配置：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

模型接口通过 OpenAI Python 软件开发工具包（SDK）调用。当前请求使用 DeepSeek 的 `thinking` 参数，服务需要支持 `tools`。`.env` 已加入 Git 忽略规则。

## 运行示例

以下命令均在项目根目录的 PowerShell 中执行，运行时会调用配置的模型服务。

首次创建示例容器并准备文件：

```powershell
docker run -d --name mini-swe-learning -w /workspace python:3.14-slim sleep infinity
docker exec -w /workspace mini-swe-learning bash -c 'echo Hello Agent! > hello.txt'
```

容器运行后启动智能体：

```powershell
uv run python -m swe.mini_agent
```

默认任务列出工作目录并读取 `hello.txt`。任务位于 `src/swe/mini_agent.py` 的 `if __name__ == "__main__":` 中；演示通过 `budget.run_with_budget()` 执行，预算在 `src/swe/budget.py` 中设置。轨迹与统计写入本地 `archive/demo/`，结束后停止本题容器。

已有容器停止后可通过 `docker start mini-swe-learning` 再次启动。

## 测试

保留的离线测试覆盖命令输出限制、镜像工具路径，以及评测预算校验、补丁回放、校准和报告汇总；不调用模型或启动真实评测容器：

```powershell
uv run python -m unittest discover -s tests -v
uv run python -m unittest discover -s evaluation -v
```

## 独立评测

`evaluation/` 与主线分离，通过命令行和 JSON 产物衔接；工作流均由 GitHub 网页端手动触发，推送代码不会自动开始评测。

- **SWE-bench Pro V2 小规模判分校准**：填写官方 `v2/tasks/` 下的单题目录名 `instance_id`，分别验证空补丁失败、参考补丁通过。不需要模型密钥，产物名为 `swebench-calibration`。
- **SWE-bench Pro V2 全量评测**：每轮固定 642 题，每题解题一次，在新容器中判分。需配置仓库机密 `DEEPSEEK_API_KEY`，最终报告产物名为 `swebench-report`。
- **SWE-bench Pro V2 全量评测（最高思考）**：独立手动入口 `evaluate_all_max.yml`，固定启用 `thinking.type=enabled` 和 `reasoning_effort=max`，使用相同密钥、模型、题目、预算及 120 秒客户端超时。最终报告产物名为 `swebench-report-max`。

校准入口为 `evaluation/calibrate.py`，全量入口为 `evaluation/swebench.py`。两者只共用评测层的 `verifier.py`，校准结果不计入全量成绩。报告包含思考模式、每题用时、词元消耗、费用、轮数、是否做对以及总成本；判分完整时发布通过率，用量完整时发布完整总费用。

最高思考入口复用全量调度，每轮单独生成清单和产物；普通全量入口始终默认关闭思考。带工具调用的思考回复会完整保留并回传 `reasoning_content`，费用仍按服务端 `usage` 中的总输出词元计算，不重复加算思考词元。

本地 `archive/`、`.env`、虚拟环境和编辑器配置均不纳入版本控制。

## 评测报告

- [2026-11-15：关闭思考模式的全量评测报告](https://github.com/DeianHsu/demo-mini-swe-agent/blob/master/reports/43650570228/report.md)
