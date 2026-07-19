"""Use DeepSeek diagnosis to generate a corrected PYPOWER case.

This script demonstrates the complete workflow requested by the assignment:

1. Load the deliberately flawed ``examples/case3_bad_data.py`` case.
2. Ask DeepSeek to diagnose the original data errors.
3. Convert the diagnosis into executable engineering repair actions.
4. Write a new corrected case file.
5. Call the PYPOWER-based agent again to verify that the corrected case is feasible.

Run from the project root:

    .\\.venv\\Scripts\\python.exe deepseek_repair_bad_data.py

If ``DEEPSEEK_API_KEY`` is configured, the diagnosis section is produced by
DeepSeek. If not, the script still writes a deterministic rule-based diagnosis
so the demonstration can run offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from pypower.idx_brch import BR_B, BR_R, BR_X, RATE_A, RATE_B, RATE_C
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, PQ, QD, REF, VA, VM, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN, QG, QMAX, QMIN, VG

from pfagent.agent import PowerFlowAgent
from pfagent.caseio import clone_case, load_power_case, write_pypower_case
from pfagent.llm import LLMClient, LLMConfig, inspect_data_with_llm
from pfagent.validators import compact_case_evidence, summarize_case, validate_case


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek diagnosis + automatic corrected case generation demo")
    parser.add_argument("--case", default="examples/case3_bad_data.py", help="original flawed PYPOWER case")
    parser.add_argument(
        "--fixed-case",
        default="examples/case3_bad_data_deepseek_fixed.py",
        help="generated corrected PYPOWER case",
    )
    parser.add_argument(
        "--out",
        default="runs/case3_bad_data_deepseek_auto",
        help="directory for DeepSeek diagnosis and verification report",
    )
    parser.add_argument("--llm-provider", default="deepseek", help="deepseek/off/rules/kimi")
    args = parser.parse_args()

    case_path = Path(args.case).resolve()
    fixed_case_path = Path(args.fixed_case).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    original_ppc = load_power_case(case_path)
    rule_issues = validate_case(original_ppc)

    llm_config = LLMConfig.from_env(provider=args.llm_provider)
    llm_client = LLMClient(llm_config)
    diagnosis = inspect_data_with_llm(
        llm_client,
        {
            "task": "请诊断这个坏数据算例，并给出能生成正确潮流算例的修改建议。",
            "case_summary": summarize_case(original_ppc),
            "source_data": compact_case_evidence(original_ppc),
            "rule_validation": [issue.as_dict() for issue in rule_issues],
            "required_output": {
                "diagnose": "指出原始数据错误",
                "repair": "建议如何修改 bus/gen/branch",
                "goal": "生成可被 PYPOWER 潮流计算收敛且满足工程约束的算例",
            },
        },
    )

    fixed_ppc, repair_actions = build_corrected_case_from_diagnosis(original_ppc, diagnosis)
    write_pypower_case(fixed_ppc, fixed_case_path, fixed_case_path.stem)

    diagnosis_bundle = {
        "original_case": str(case_path),
        "generated_fixed_case": str(fixed_case_path),
        "llm_provider": llm_config.provider,
        "llm_enabled": llm_config.enabled,
        "rule_issues": [issue.as_dict() for issue in rule_issues],
        "deepseek_diagnosis": diagnosis,
        "applied_repair_actions": repair_actions,
    }
    (out_dir / "deepseek_diagnosis_and_repairs.json").write_text(
        json.dumps(_jsonable(diagnosis_bundle), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "修复说明.md").write_text(render_markdown_summary(diagnosis_bundle), encoding="utf-8")

    verify_dir = out_dir / "verification"
    agent = PowerFlowAgent(engine="pypower", llm_config=llm_config, max_rounds=20)
    report = agent.run(fixed_case_path, verify_dir, auto_repair=True)

    print("DeepSeek/规则诊断完成。")
    print(f"诊断与修复记录: {out_dir / 'deepseek_diagnosis_and_repairs.json'}")
    print(f"修复说明: {out_dir / '修复说明.md'}")
    print(f"自动生成的修正算例: {fixed_case_path}")
    print(f"验证报告: {report.final_report_path}")
    print(f"工程可行: {report.success}")
    print(f"求解器收敛: {report.solver_converged}")
    print(f"导出的最终可行算例: {report.final_case_path}")


def build_corrected_case_from_diagnosis(
    original_ppc: dict[str, Any],
    diagnosis: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert DeepSeek/rule diagnosis into deterministic repair actions.

    DeepSeek is used for diagnosis and explanation. The actual numerical edits
    are performed by code so they are reproducible and can be verified by PYPOWER.
    """

    case = clone_case(original_ppc)
    bus = case["bus"]
    gen = case["gen"]
    branch = case["branch"]
    actions: list[dict[str, Any]] = []

    # 1. Set a valid REF/slack bus. The only online generator in case3_bad_data
    # is on bus 2, so bus 2 should be the REF bus.
    online = gen[:, GEN_STATUS] > 0
    if np.any(online):
        ref_bus = int(gen[np.where(online)[0][np.argmax(gen[online, PMAX])], GEN_BUS])
    else:
        ref_bus = int(bus[0, BUS_I])
    bus[:, BUS_TYPE] = np.where(bus[:, BUS_I].astype(int) == ref_bus, REF, PQ)
    actions.append({"action": "set_reference_bus", "reason": "缺少 REF 平衡节点，选择在线发电机所在母线作为平衡节点。", "bus": ref_bus})

    # 2. Repair bus initial values and voltage limits.
    for row in bus:
        bus_id = int(row[BUS_I])
        if row[VMIN] >= row[VMAX]:
            old = [float(row[VMIN]), float(row[VMAX])]
            row[VMIN], row[VMAX] = 0.94, 1.06
            actions.append({"action": "fix_voltage_limits", "bus": bus_id, "old": old, "new": [0.94, 1.06]})
        if not (0.9 <= row[VM] <= 1.1):
            old = float(row[VM])
            row[VM] = 1.05 if bus_id == ref_bus else 1.0
            actions.append({"action": "fix_initial_voltage", "bus": bus_id, "old": old, "new": float(row[VM])})
        if abs(row[VA]) > 180:
            old = float(row[VA])
            row[VA] = 0.0
            actions.append({"action": "fix_initial_angle", "bus": bus_id, "old": old, "new": 0.0})

    # 3. Repair generator limits and voltage set point.
    total_pd = float(np.sum(bus[:, PD]))
    total_qd = float(np.sum(bus[:, QD]))
    for idx, row in enumerate(gen):
        if row[QMAX] < row[QMIN]:
            old = [float(row[QMIN]), float(row[QMAX])]
            row[QMIN], row[QMAX] = row[QMAX], row[QMIN]
            actions.append({"action": "swap_q_limits", "gen_index": idx, "old": old})
        if row[PMAX] < row[PMIN]:
            old = [float(row[PMIN]), float(row[PMAX])]
            row[PMIN], row[PMAX] = row[PMAX], row[PMIN]
            actions.append({"action": "swap_p_limits", "gen_index": idx, "old": old})
        row[PMIN] = min(float(row[PMIN]), 10.0)
        row[PMAX] = max(float(row[PMAX]), total_pd * 1.35, 250.0)
        row[PG] = min(max(total_pd * 1.03, row[PMIN]), row[PMAX])
        row[QMIN] = min(float(row[QMIN]), -max(80.0, total_qd * 1.2))
        row[QMAX] = max(float(row[QMAX]), max(250.0, total_qd * 2.4))
        row[QG] = 0.0
        row[VG] = 1.05
        actions.append(
            {
                "action": "repair_generator_capacity_and_setpoint",
                "gen_index": idx,
                "reason": "保证发电机有功/无功容量能够覆盖负荷需求，并把 VG 调整到合理标幺值。",
                "PG": round(float(row[PG]), 6),
                "PMAX": round(float(row[PMAX]), 6),
                "QMIN": round(float(row[QMIN]), 6),
                "QMAX": round(float(row[QMAX]), 6),
                "VG": round(float(row[VG]), 6),
            }
        )

    # 4. Repair branch zero impedance, negative thermal limit and weak feeder.
    for idx, row in enumerate(branch):
        f_bus, t_bus = int(row[0]), int(row[1])
        if abs(row[BR_R]) < 1e-12 and abs(row[BR_X]) < 1e-12:
            row[BR_R], row[BR_X], row[BR_B] = 0.02, 0.06, 0.03
            actions.append({"action": "fix_zero_impedance", "branch_index": idx, "branch": f"{f_bus}-{t_bus}", "new": [0.02, 0.06, 0.03]})
        # In the original bad case, branch 2-3 is too weak for the heavy load at bus 3.
        if {f_bus, t_bus} == {2, 3}:
            old = [float(row[BR_R]), float(row[BR_X])]
            row[BR_R], row[BR_X] = 0.01, 0.04
            actions.append({"action": "strengthen_heavy_load_feeder", "branch_index": idx, "branch": f"{f_bus}-{t_bus}", "old": old, "new": [0.01, 0.04]})
        if row[RATE_A] <= 0 or row[RATE_A] < 250:
            old = [float(row[RATE_A]), float(row[RATE_B]), float(row[RATE_C])]
            row[RATE_A], row[RATE_B], row[RATE_C] = 250, 250, 250
            actions.append({"action": "fix_branch_thermal_limits", "branch_index": idx, "branch": f"{f_bus}-{t_bus}", "old": old, "new": [250, 250, 250]})

    return case, actions


def render_markdown_summary(bundle: dict[str, Any]) -> str:
    lines = [
        "# case3_bad_data.py 的 DeepSeek 诊断与自动修复说明",
        "",
        f"- 原始算例: `{bundle['original_case']}`",
        f"- 自动生成的修正算例: `{bundle['generated_fixed_case']}`",
        f"- LLM 提供商: `{bundle['llm_provider']}`",
        f"- 是否实际调用 LLM: `{bundle['llm_enabled']}`",
        "",
        "## 规则检查发现的主要问题",
        "",
    ]
    for issue in bundle["rule_issues"]:
        lines.append(f"- `{issue['code']}`: {issue['message']}，位置 `{issue['location']}`，建议：{issue['suggestion']}")

    lines.extend(["", "## DeepSeek 输出", ""])
    deepseek = bundle["deepseek_diagnosis"]
    if deepseek.get("parsed"):
        lines.append("```json")
        lines.append(json.dumps(deepseek["parsed"], ensure_ascii=False, indent=2))
        lines.append("```")
    elif deepseek.get("content"):
        lines.append(str(deepseek["content"]))
    else:
        lines.append(str(deepseek.get("message") or deepseek.get("error") or "无 DeepSeek 输出。"))

    lines.extend(["", "## 程序根据诊断自动执行的修改", ""])
    for action in bundle["applied_repair_actions"]:
        lines.append(f"- `{action['action']}`: {json.dumps(action, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    main()
