# 电力系统潮流计算智能体（PYPOWER + DeepSeek/Kimi）

这是一个可在 VS Code 中直接演示的课程任务版本。程序以 PYPOWER 为确定性计算核心，完成：

1. 读取 MATPOWER/PYPOWER 风格数据并执行潮流计算；
2. 检查原始数据、节点类型、孤岛、机组限额、支路参数和工程约束；
3. 识别 PV/REF 发电机无功越限，并按 Q 限额执行 PV→PQ 处理；
4. 对不收敛或不可行算例尝试平坦启动、算法切换、机组再调度和基准负荷搜索；
5. 可选调用 DeepSeek、Kimi 或其他 OpenAI-compatible 模型分析数据、过程和结果；
6. 生成 Markdown 报告、JSON 报告和可复算的 Python 算例。

> 核心原则：大语言模型负责诊断和提出候选措施，PYPOWER 负责数值计算与最终验证。只有重新计算收敛且满足工程约束的方案，才会被导出为最终可行算例。

## 一、VS Code 一键运行

### 1. 准备软件

- Python 3.10～3.12
- VS Code
- VS Code 的 Microsoft Python 扩展

### 2. 打开项目

在 VS Code 中选择“文件 → 打开文件夹”，打开本项目根目录。

### 3. 安装环境

按 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，运行：

```text
1. 安装运行环境
```

脚本会创建 `.venv` 并安装与 PYPOWER 兼容的 NumPy/SciPy 版本。

### 4. 运行演示

再次选择 `Tasks: Run Task`，运行：

```text
2. 运行完整演示
```

也可以按 `F5`，选择：

```text
运行潮流智能体完整演示
```

演示结果位于 `runs` 目录。

## 二、手动命令

```powershell
.\.venv\Scripts\python.exe -m pfagent doctor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pfagent inspect .\examples\case3_bad_data.py
.\.venv\Scripts\python.exe -m pfagent run .\examples\case9_demo.py --engine pypower
.\.venv\Scripts\python.exe -m pfagent run .\examples\case3_q_limit.py --engine pypower --max-rounds 20
```

## 三、DeepSeek 自动诊断并生成修正算例

先在 PowerShell 当前会话中设置 API Key（不要把真实 Key 写入代码）：

```powershell
$env:DEEPSEEK_API_KEY="你的API Key"
```

然后执行完整闭环：读取坏数据、调用 DeepSeek 诊断、自动生成修正文件，并用 PYPOWER 重新验证：

```powershell
.\.venv\Scripts\python.exe .\deepseek_repair_bad_data.py `
  --case .\examples\case3_bad_data.py `
  --fixed-case .\examples\case3_bad_data_deepseek_fixed.py `
  --out .\runs\bad_data_deepseek_auto `
  --llm-provider deepseek
```

重点输出：

- `runs/bad_data_deepseek_auto/修复说明.md`：DeepSeek 诊断和修复说明；
- `examples/case3_bad_data_deepseek_fixed.py`：自动生成的修正算例；
- `runs/bad_data_deepseek_auto/verification/report.md`：PYPOWER 验证报告；
- `runs/bad_data_deepseek_auto/verification/final_feasible_case.py`：最终可行运行方式。

## 四、演示算例

- `case9_demo.py`：标准 IEEE 9 节点算例，展示正常潮流。
- `case3_repairable.py`：缺少 REF 且初值异常，展示数据检查和修复。
- `case3_q_limit.py`：PV 节点 Q 上限很紧，展示 Q 越限、PV→PQ 和可行方案搜索。
- `case3_bad_data.py`：包含多种错误且物理能力不足，展示“数据修复后仍可能无解”。

## 五、DeepSeek

API Key 不要写入代码。在 VS Code PowerShell 终端中设置：

```powershell
$env:DEEPSEEK_API_KEY="你的API Key"
```

然后运行：

```powershell
.\.venv\Scripts\python.exe -m pfagent run .\examples\case3_q_limit.py `
  --engine pypower `
  --llm-provider deepseek `
  --max-rounds 20 `
  --out .\runs\deepseek_q_limit
```

## 六、Kimi

```powershell
$env:KIMI_API_KEY="你的API Key"
.\.venv\Scripts\python.exe -m pfagent run .\examples\case3_q_limit.py `
  --engine pypower `
  --llm-provider kimi `
  --max-rounds 20 `
  --out .\runs\kimi_q_limit
```

如果服务商更新了模型名称，可增加：

```text
--llm-model 当前可用模型名
```

## 七、代码结构

```text
demo_vscode.py          VS Code 完整演示入口
deepseek_repair_bad_data.py  DeepSeek 坏数据诊断、修复与验证流程
examples/               潮流算例
pfagent/agent.py        智能体闭环控制
pfagent/caseio.py       数据读取与导出
pfagent/validators.py   数据和工程约束检查
pfagent/solvers.py      PYPOWER/MATPOWER 求解器与雅可比诊断
pfagent/repairs.py      自动修复动作
pfagent/llm.py          DeepSeek/Kimi 接口
pfagent/reporting.py    Markdown/JSON 报告
tests/                  自动测试
.vscode/                VS Code 任务和调试配置
```

## 八、结果状态

程序严格区分：

- `success=True`：求解器收敛，且没有电压、线路或阻断性 Q 越限；
- `solver_converged=True`：潮流方程数值收敛，但不一定工程可行。

只有工程可行时才输出：

```text
final_feasible_case.py
```

若只是数值收敛但存在越限，则输出：

```text
last_converged_but_infeasible_case.py
```

## 九、仓库安全说明

- 不要提交 `.env`、真实 API Key、`.venv/` 或 `runs/`；
- 仓库中的 `.env.example` 仅列出变量名称，不包含密钥；
- GitHub Actions 会在每次推送和拉取请求时自动运行测试。

## 十、核心表述

大语言模型不替代潮流求解器。PYPOWER 负责数值求解，规则模块负责确定性校验，LLM 根据结构化数据、工程越限和雅可比数值诊断模仿工程师提出解释与候选措施。所有候选措施必须重新通过 PYPOWER 计算和工程约束复核后，才能被判定为可行运行方式。
