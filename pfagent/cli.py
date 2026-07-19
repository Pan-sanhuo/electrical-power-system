from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np

from .agent import PowerFlowAgent
from .llm import LLMConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pfagent", description="电力系统潮流计算智能体")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="执行潮流计算、诊断与自动修复")
    run_p.add_argument("case", help="MATPOWER .m/.mat、PYPOWER .py 或 JSON 算例文件")
    run_p.add_argument("--engine", choices=["pypower", "matpower"], default="pypower")
    run_p.add_argument("--out", default=None, help="输出目录，默认 runs/<case>_<timestamp>")
    run_p.add_argument("--max-rounds", type=int, default=20)
    run_p.add_argument("--no-auto-repair", action="store_true", help="只计算和诊断，不执行自动修复")
    run_p.add_argument("--matpower-path", default=os.getenv("MATPOWER_PATH"), help="MATPOWER 根目录")
    _add_llm_args(run_p)

    inspect_p = sub.add_parser("inspect", help="只检查原始算例数据")
    inspect_p.add_argument("case")
    _add_llm_args(inspect_p)

    sub.add_parser("doctor", help="检查本机依赖与外部求解器环境")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        _doctor()
        return

    llm_config = LLMConfig.from_env(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
    )
    agent = PowerFlowAgent(
        engine=getattr(args, "engine", "pypower"),
        llm_config=llm_config,
        matpower_path=getattr(args, "matpower_path", None),
        max_rounds=getattr(args, "max_rounds", 20),
    )

    if args.command == "inspect":
        summary, issues, llm = agent.inspect(args.case)
        print("算例摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("\n规则检查:")
        if issues:
            for issue in issues:
                print(f"  [{issue['level']}] {issue['code']} {issue['location']}: {issue['message']}；建议：{issue['suggestion']}")
        else:
            print("  未发现问题")
        if llm.get("enabled") is False:
            print(f"\nLLM: {llm['message']}")
        elif llm.get("error"):
            print(f"\nLLM 调用失败: {llm['error']}")
        else:
            print(f"\nLLM 输出: {llm.get('parsed') or llm.get('content')}")
        return

    report = agent.run(args.case, args.out, auto_repair=not args.no_auto_repair)
    print(f"最终状态: {'已获得工程可行潮流' if report.success else '未获得通过全部约束的可行潮流'}")
    print(f"求解器数值收敛: {report.solver_converged}")
    print(f"报告: {report.final_report_path}")
    print(f"JSON: {report.final_json_path}")
    print(f"算例: {report.final_case_path}")
    if report.attempts:
        last = report.attempts[-1]
        print(f"最后一次尝试: {last.name}, converged={last.success}, feasible={last.feasible}, error={last.error}")


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-provider", default="off", help="off/deepseek/kimi/openai-compatible 名称")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None, help="也可用环境变量 DEEPSEEK_API_KEY、KIMI_API_KEY/MOONSHOT_API_KEY 等")


def _doctor() -> None:
    print("Python 依赖:")
    for name in ("numpy", "scipy", "pypower"):
        spec = importlib.util.find_spec(name)
        print(f"  {name}: {'OK' if spec else 'MISSING'}")
    print(f"  numpy version: {np.__version__}")
    print("\n外部程序:")
    print(f"  matlab: {shutil.which('matlab') or '未找到'}")
    matpower_path = os.getenv("MATPOWER_PATH")
    print(f"  MATPOWER_PATH: {matpower_path or '未设置'}")
    if matpower_path:
        print(f"  MATPOWER_PATH exists: {Path(matpower_path).exists()}")


if __name__ == "__main__":
    main()
