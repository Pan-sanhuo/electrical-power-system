"""VS Code demonstration entry point for the power-flow agent."""

import os
from pathlib import Path

from pfagent import LLMConfig, PowerFlowAgent


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def llm_config_from_environment() -> LLMConfig:
    if os.getenv("DEEPSEEK_API_KEY"):
        print("已检测到 DEEPSEEK_API_KEY：演示将调用 DeepSeek。")
        return LLMConfig.from_env(provider="deepseek")
    if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
        print("已检测到 KIMI/MOONSHOT API Key：演示将调用 Kimi。")
        return LLMConfig.from_env(provider="kimi")
    return LLMConfig(provider="off")


def run_demo(title: str, case_name: str, output_name: str, llm_config: LLMConfig, max_rounds: int = 20) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    agent = PowerFlowAgent(
        engine="pypower",
        llm_config=llm_config,
        max_rounds=max_rounds,
    )
    report = agent.run(ROOT / "examples" / case_name, RUNS / output_name)
    print(f"工程可行: {report.success}")
    print(f"求解器收敛: {report.solver_converged}")
    print(f"计算尝试次数: {len(report.attempts)}")
    print(f"自动修复动作数: {len(report.repairs)}")
    print(f"报告: {report.final_report_path}")
    print(f"结构化结果: {report.final_json_path}")
    print(f"导出算例: {report.final_case_path}")
    if report.attempts:
        last = report.attempts[-1]
        print(f"最后方案: {last.name}")
        print(f"工程越限数: {len(last.violations)}")
        print(f"Q限额事件数: {len(last.q_limit_events)}")
        print(f"雅可比初始诊断: {last.diagnostics}")


def main() -> None:
    print("电力系统潮流计算智能体：PYPOWER 完整演示")
    print("说明：未配置 API Key 时使用确定性规则；配置后可启用 DeepSeek/Kimi。")
    llm_config = llm_config_from_environment()
    run_demo("演示一：标准 IEEE 9 节点潮流", "case9_demo.py", "demo_case9", llm_config)
    run_demo("演示二：原始数据自动检查与修复", "case3_repairable.py", "demo_repairable", llm_config)
    run_demo("演示三：PV 无功越限、PV→PQ 与可行方案搜索", "case3_q_limit.py", "demo_q_limit", llm_config)
    print("\n演示完成。请在 runs 目录打开各算例的 report.md。")


if __name__ == "__main__":
    main()
