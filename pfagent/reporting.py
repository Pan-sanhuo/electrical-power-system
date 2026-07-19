from __future__ import annotations

from pathlib import Path
from typing import Any

from .caseio import write_json_report
from .models import AgentRunReport
from .validators import summarize_result


def write_markdown_report(report: AgentRunReport) -> Path:
    out = report.output_dir / "report.md"
    lines: list[str] = []
    lines.append("# 电力系统潮流计算智能体报告")
    lines.append("")
    lines.append(f"- 算例文件: `{report.case_path}`")
    lines.append(f"- 输出目录: `{report.output_dir}`")
    lines.append(f"- 最终状态: {'已获得工程约束内的可行潮流' if report.success else '未获得通过全部工程约束的可行潮流'}")
    lines.append(f"- 最后一次求解器数值收敛: {report.solver_converged}")
    if report.final_case_path:
        lines.append(f"- 导出算例: `{report.final_case_path}`")

    lines.append("")
    lines.append("## 1. 原始数据检查")
    if report.validation:
        lines.append("|级别|代码|位置|问题|建议|")
        lines.append("|---|---|---|---|---|")
        for issue in report.validation:
            lines.append(f"|{issue.level}|{issue.code}|`{issue.location}`|{issue.message}|{issue.suggestion}|")
    else:
        lines.append("未发现规则校验问题。")
    _append_llm_section(lines, "LLM 原始数据复核", report.llm_sections.get("data_inspection"))

    lines.append("")
    lines.append("## 2. 计算尝试")
    lines.append("|序号|尝试|求解器|算法|无功限额|结果|耗时(s)|问题|")
    lines.append("|---:|---|---|---|---|---|---:|---|")
    for idx, attempt in enumerate(report.attempts, 1):
        options = attempt.options
        issue_count = len(attempt.violations) + len(attempt.q_limit_events)
        status = "可行" if attempt.feasible else ("数值收敛但不可行" if attempt.success else "未收敛")
        if issue_count:
            status += f"，{issue_count} 项风险"
        lines.append(
            f"|{idx}|{attempt.name}|{attempt.engine}|{options.get('pf_alg_name', options.get('pf_alg'))}|"
            f"{options.get('enforce_q_lims')}|{status}|{attempt.elapsed_s:.4f}|{attempt.error or ''}|"
        )
        if attempt.q_limit_events:
            lines.append("")
            lines.append(f"尝试 {idx} 的无功限额事件：")
            for event in attempt.q_limit_events:
                lines.append(f"- {event.get('type')}: bus {event.get('bus')}, Q={event.get('qg')}, limit={event.get('limit')}。{event.get('message')}")
        if attempt.violations:
            lines.append("")
            lines.append(f"尝试 {idx} 的工程越限：")
            for violation in attempt.violations:
                lines.append(f"- {violation.get('type')}: {violation.get('target')} = {violation.get('value')}，限值 {violation.get('limit')}。{violation.get('message')}")
        if attempt.diagnostics:
            lines.append("")
            lines.append(f"尝试 {idx} 的初始数值诊断：`{attempt.diagnostics}`")

    lines.append("")
    lines.append("## 3. 自动修复动作")
    if report.repairs:
        for action in report.repairs:
            lines.append(f"- `{action.action}`: {action.reason} {action.details}")
    else:
        lines.append("未执行自动修复动作。")

    final = next((attempt for attempt in reversed(report.attempts) if attempt.feasible and attempt.result is not None), None)
    if final is None:
        final = next((attempt for attempt in reversed(report.attempts) if attempt.result is not None), None)
    if final and final.result:
        lines.append("")
        lines.append("## 4. 最终结果摘要")
        summary = summarize_result(final.result)
        for key, value in summary.items():
            lines.append(f"- {key}: {value}")
    _append_llm_section(lines, "LLM 结果诊断", report.llm_sections.get("result_diagnosis"))
    _append_llm_section(lines, "LLM 修复建议", report.llm_sections.get("repair_proposal"))

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_report_bundle(report: AgentRunReport) -> AgentRunReport:
    report.output_dir.mkdir(parents=True, exist_ok=True)
    report.final_report_path = write_markdown_report(report)
    report.final_json_path = (report.output_dir / "report.json").resolve()
    write_json_report(report.as_dict(), report.final_json_path)
    return report


def _append_llm_section(lines: list[str], title: str, section: Any) -> None:
    lines.append("")
    lines.append(f"### {title}")
    if not section:
        lines.append("无。")
        return
    if isinstance(section, dict) and section.get("enabled") is False:
        lines.append(section.get("message", "LLM 未启用。"))
        return
    if isinstance(section, dict) and section.get("error"):
        lines.append(f"LLM 调用失败: {section['error']}")
        return
    parsed = section.get("parsed") if isinstance(section, dict) else None
    if parsed is not None:
        lines.append("```json")
        import json

        lines.append(json.dumps(parsed, ensure_ascii=False, indent=2))
        lines.append("```")
    elif isinstance(section, dict) and section.get("content"):
        lines.append(section["content"])
    else:
        lines.append("无结构化输出。")
