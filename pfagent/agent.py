from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .caseio import clone_case, load_power_case, write_pypower_case
from .llm import LLMClient, LLMConfig, diagnose_result_with_llm, inspect_data_with_llm, propose_repairs_with_llm
from .models import AgentRunReport, RepairAction, SolveAttempt
from .repairs import apply_flat_start, redispatch_generation, repair_obvious_data_errors, scale_load, tune_voltage_setpoints
from .reporting import write_report_bundle
from .solvers import MatpowerSolver, PowerFlowOptions, PyPowerSolver
from .validators import (
    compact_case_evidence,
    has_blocking_validation_errors,
    operating_violations,
    q_limit_violations,
    summarize_case,
    summarize_result,
    validate_case,
)


class PowerFlowAgent:
    def __init__(
        self,
        engine: str = "pypower",
        llm_config: LLMConfig | None = None,
        matpower_path: str | None = None,
        max_rounds: int = 20,
    ):
        self.engine = engine.lower()
        self.llm = LLMClient(llm_config or LLMConfig())
        self.matpower_path = matpower_path or os.getenv("MATPOWER_PATH")
        self.max_rounds = max_rounds

    def run(self, case_path: str | Path, output_dir: str | Path | None = None, auto_repair: bool = True) -> AgentRunReport:
        case_file = Path(case_path).resolve()
        out_dir = self._output_dir(case_file, output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        original_ppc = load_power_case(case_file)
        working_ppc = clone_case(original_ppc)
        validation = validate_case(original_ppc)
        repairs: list[RepairAction] = []
        attempts: list[SolveAttempt] = []
        llm_sections: dict[str, Any] = {}

        llm_sections["data_inspection"] = inspect_data_with_llm(
            self.llm,
            {
                "case_summary": summarize_case(original_ppc),
                "source_data": compact_case_evidence(original_ppc),
                "rule_validation": [issue.as_dict() for issue in validation],
            },
        )

        unrecoverable_codes = {"BASE_MVA", "SHAPE", "NAN_INF", "DUP_BUS", "BUS_TYPE", "GEN_BUS", "BRANCH_BUS"}
        has_unrecoverable_data = any(
            issue.level == "error" and issue.code in unrecoverable_codes for issue in validation
        )
        if auto_repair and not has_unrecoverable_data:
            working_ppc, data_repairs = repair_obvious_data_errors(working_ppc)
            repairs.extend(data_repairs)
        post_repair_validation = validate_case(working_ppc)

        def attempt(name: str, ppc: dict[str, Any], options: PowerFlowOptions) -> SolveAttempt:
            if len(attempts) >= self.max_rounds:
                return attempts[-1]
            result = self._solve(ppc, options, out_dir)
            violations: list[dict[str, Any]] = []
            q_events: list[dict[str, Any]] = []
            if result.ppc is not None:
                violations = operating_violations(result.ppc)
                q_events = q_limit_violations(ppc, result.ppc)
            item = SolveAttempt(
                name=name,
                engine=options.engine,
                options=options.as_dict(),
                success=result.success,
                elapsed_s=result.elapsed_s,
                result=result.ppc,
                error=result.error,
                violations=violations,
                q_limit_events=q_events,
                diagnostics=(result.raw or {}).get("diagnostics", {}),
            )
            attempts.append(item)
            return item
        attempt.can_run = lambda: len(attempts) < self.max_rounds

        if has_blocking_validation_errors(post_repair_validation):
            llm_sections["result_diagnosis"] = diagnose_result_with_llm(
                self.llm,
                {
                    "case_summary": summarize_case(working_ppc),
                    "blocking_validation": [issue.as_dict() for issue in post_repair_validation],
                    "final_success": False,
                },
            )
            final_case_path = write_pypower_case(working_ppc, out_dir / "last_modified_case.py", "last_modified_case")
            report = AgentRunReport(
                case_path=case_file,
                output_dir=out_dir,
                validation=validation,
                attempts=attempts,
                repairs=repairs,
                llm_sections=llm_sections,
                final_case_path=final_case_path,
            )
            return write_report_bundle(report)

        base_options = self._options(pf_alg=1, enforce_q_lims=False)
        current = attempt("screen_nr_without_q_limits", working_ppc, base_options)

        if auto_repair and not self._is_clean(current):
            llm_sections["repair_proposal"] = propose_repairs_with_llm(
                self.llm,
                {
                    "allowed_actions": [
                        {"action": "enable_q_limits"},
                        {"action": "flat_start"},
                        {"action": "try_fdxb"},
                        {"action": "try_fdbx"},
                        {"action": "try_gauss_seidel"},
                        {"action": "redispatch_generation"},
                        {"action": "scale_load", "factor": "0.80-0.99"},
                        {"action": "repair_data_hygiene"},
                    ],
                    "case_summary": summarize_case(working_ppc),
                    "validation": [issue.as_dict() for issue in validation],
                    "attempts": [item.as_dict() for item in attempts],
                },
            )
            proposal = llm_sections["repair_proposal"]
            if proposal.get("enabled") and proposal.get("parsed"):
                current, working_ppc = self._try_llm_actions(proposal, working_ppc, repairs, attempt)

        if auto_repair and not self._is_clean(current):
            current, working_ppc = self._deterministic_repair_sequence(working_ppc, repairs, attempt, current)

        if auto_repair and current.success and not self._is_operationally_clean(current):
            current, working_ppc = self._post_convergence_repairs(working_ppc, repairs, attempt, current)

        final_attempt = next((item for item in reversed(attempts) if item.feasible and item.result is not None), None)
        last_result_attempt = next((item for item in reversed(attempts) if item.result is not None), None)
        final_result = final_attempt.result if final_attempt else (last_result_attempt.result if last_result_attempt else None)
        final_case_path = None
        if final_attempt is not None and final_result is not None:
            final_case_path = write_pypower_case(final_result, out_dir / "final_feasible_case.py", "final_feasible_case")
        elif final_result is not None:
            final_case_path = write_pypower_case(
                final_result,
                out_dir / "last_converged_but_infeasible_case.py",
                "last_converged_but_infeasible_case",
            )
        elif working_ppc is not None:
            final_case_path = write_pypower_case(working_ppc, out_dir / "last_modified_case.py", "last_modified_case")

        llm_sections["result_diagnosis"] = diagnose_result_with_llm(
            self.llm,
            {
                "case_summary": summarize_case(working_ppc),
                "attempts": [item.as_dict() for item in attempts],
                "final_summary": summarize_result(final_result) if final_result is not None else None,
                "final_success": bool(attempts and attempts[-1].feasible),
                "final_solver_converged": bool(attempts and attempts[-1].success),
                "final_violations": attempts[-1].violations if attempts else [],
                "final_q_limit_events": attempts[-1].q_limit_events if attempts else [],
                "numerical_diagnostics": attempts[-1].diagnostics if attempts else {},
            },
        )

        report = AgentRunReport(
            case_path=case_file,
            output_dir=out_dir,
            validation=validation,
            attempts=attempts,
            repairs=repairs,
            llm_sections=llm_sections,
            final_case_path=final_case_path,
        )
        return write_report_bundle(report)

    def inspect(self, case_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        ppc = load_power_case(case_path)
        issues = validate_case(ppc)
        llm = inspect_data_with_llm(
            self.llm,
            {
                "case_summary": summarize_case(ppc),
                "source_data": compact_case_evidence(ppc),
                "rule_validation": [i.as_dict() for i in issues],
            },
        )
        return summarize_case(ppc), [issue.as_dict() for issue in issues], llm

    def _solve(self, ppc: dict[str, Any], options: PowerFlowOptions, out_dir: Path):
        if self.engine == "matpower":
            return MatpowerSolver().run(ppc, options, out_dir)
        return PyPowerSolver().run(ppc, options)

    def _options(self, pf_alg: int, enforce_q_lims: bool) -> PowerFlowOptions:
        return PowerFlowOptions(
            engine=self.engine,
            pf_alg=pf_alg,
            enforce_q_lims=enforce_q_lims,
            max_it=30,
            matpower_path=self.matpower_path,
        )

    def _try_llm_actions(
        self,
        proposal: dict[str, Any],
        working_ppc: dict[str, Any],
        repairs: list[RepairAction],
        attempt,
    ) -> tuple[SolveAttempt, dict[str, Any]]:
        current = None
        parsed = proposal.get("parsed") if isinstance(proposal, dict) else None
        actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
        for raw in actions[:4]:
            if not self._can_attempt(attempt):
                break
            action_name = raw.get("action") if isinstance(raw, dict) else str(raw)
            if action_name == "enable_q_limits":
                repair = RepairAction("enable_q_limits", "LLM 建议启用 PV/REF 无功限额处理", {})
                repairs.append(repair)
                current = attempt("llm_nr_enforce_q_limits", working_ppc, self._options(1, True))
            elif action_name == "flat_start":
                working_ppc, repair = apply_flat_start(working_ppc)
                repair.reason = "LLM 建议：" + repair.reason
                repairs.append(repair)
                current = attempt("llm_flat_start_nr", working_ppc, self._options(1, True))
            elif action_name == "try_fdxb":
                repairs.append(RepairAction("try_fdxb", "LLM 建议切换为快速解耦 XB 算法", {}))
                current = attempt("llm_fdxb", working_ppc, self._options(2, True))
            elif action_name == "try_fdbx":
                repairs.append(RepairAction("try_fdbx", "LLM 建议切换为快速解耦 BX 算法", {}))
                current = attempt("llm_fdbx", working_ppc, self._options(3, True))
            elif action_name == "try_gauss_seidel":
                repairs.append(RepairAction("try_gauss_seidel", "LLM 建议使用 Gauss-Seidel 作为数值后备", {}))
                current = attempt("llm_gauss_seidel", working_ppc, self._options(4, True))
            elif action_name == "redispatch_generation":
                working_ppc, repair = redispatch_generation(working_ppc)
                repair.reason = "LLM 建议：" + repair.reason
                repairs.append(repair)
                current = attempt("llm_redispatch_nr", working_ppc, self._options(1, True))
            elif action_name == "scale_load":
                factor = float(raw.get("factor", 0.95)) if isinstance(raw, dict) else 0.95
                factor = min(max(factor, 0.8), 0.99)
                working_ppc, repair = scale_load(working_ppc, factor, "LLM 建议按比例调整负荷以寻找可行运行方式")
                repairs.append(repair)
                current = attempt(f"llm_scale_load_{factor:.2f}_nr", working_ppc, self._options(1, True))
            elif action_name == "repair_data_hygiene":
                working_ppc, data_repairs = repair_obvious_data_errors(working_ppc)
                repairs.extend(data_repairs)
                current = attempt("llm_data_hygiene_nr", working_ppc, self._options(1, True))

            if current is not None and self._is_clean(current):
                return current, working_ppc

        if current is None:
            current = attempt("nr_enforce_q_limits", working_ppc, self._options(1, True))
        return current, working_ppc

    def _deterministic_repair_sequence(self, working_ppc: dict[str, Any], repairs: list[RepairAction], attempt, current: SolveAttempt):
        if (self._has_blocking_q_violation(current) or not current.success) and self._can_attempt(attempt):
            repairs.append(RepairAction("enable_q_limits", "发现 PV/REF 无功越限或初始计算失败，启用 ENFORCE_Q_LIMS", {}))
            current = attempt("nr_enforce_q_limits", working_ppc, self._options(1, True))
            if self._is_clean(current):
                return current, working_ppc

        if not self._can_attempt(attempt):
            return current, working_ppc
        working_ppc, repair = apply_flat_start(working_ppc)
        repairs.append(repair)
        current = attempt("flat_start_nr", working_ppc, self._options(1, True))
        if self._is_clean(current):
            return current, working_ppc

        for alg, name, reason in (
            (2, "fdxb", "Newton-Raphson 未收敛或数值病态，尝试快速解耦 XB 算法"),
            (3, "fdbx", "继续尝试快速解耦 BX 算法"),
            (4, "gauss_seidel", "使用 Gauss-Seidel 作为鲁棒性后备算法"),
        ):
            if not self._can_attempt(attempt):
                return current, working_ppc
            repairs.append(RepairAction(f"try_{name}", reason, {"pf_alg": alg}))
            current = attempt(name, working_ppc, self._options(alg, True))
            if self._is_clean(current):
                return current, working_ppc

        if not self._can_attempt(attempt):
            return current, working_ppc
        working_ppc, repair = redispatch_generation(working_ppc)
        repairs.append(repair)
        current = attempt("redispatch_nr", working_ppc, self._options(1, True))
        if self._is_clean(current):
            return current, working_ppc

        load_search_base = clone_case(working_ppc)
        for factor in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65):
            if not self._can_attempt(attempt):
                return current, working_ppc
            candidate_ppc, repair = scale_load(load_search_base, factor, "以同一基准算例按比例削减负荷，搜索可行运行方式")
            repairs.append(repair)
            current = attempt(f"scale_load_{factor:.2f}_nr", candidate_ppc, self._options(1, True))
            if self._is_clean(current):
                return current, candidate_ppc
        return current, working_ppc

    def _post_convergence_repairs(self, working_ppc: dict[str, Any], repairs: list[RepairAction], attempt, current: SolveAttempt):
        for _ in range(3):
            if not self._can_attempt(attempt):
                return current, working_ppc
            types = {v["type"] for v in current.violations}
            if "LOW_VOLTAGE" in types:
                working_ppc, repair = tune_voltage_setpoints(working_ppc, "raise")
                repairs.append(repair)
                current = attempt("raise_voltage_setpoints_nr", working_ppc, self._options(1, True))
            elif "HIGH_VOLTAGE" in types:
                working_ppc, repair = tune_voltage_setpoints(working_ppc, "lower")
                repairs.append(repair)
                current = attempt("lower_voltage_setpoints_nr", working_ppc, self._options(1, True))
            elif "BRANCH_OVERLOAD" in types:
                working_ppc, repair = scale_load(working_ppc, 0.95, "收敛但存在线路过载，先按 5% 负荷削减形成可行运行方式")
                repairs.append(repair)
                current = attempt("post_overload_scale_load_nr", working_ppc, self._options(1, True))
            else:
                return current, working_ppc
            if self._is_operationally_clean(current):
                return current, working_ppc
        return current, working_ppc

    def _is_clean(self, attempt: SolveAttempt) -> bool:
        return attempt.success and self._is_operationally_clean(attempt)

    def _is_operationally_clean(self, attempt: SolveAttempt) -> bool:
        return attempt.success and not attempt.violations and not self._has_blocking_q_violation(attempt)

    def _has_blocking_q_violation(self, attempt: SolveAttempt) -> bool:
        return any(event.get("type") in {"Q_MAX_VIOLATION", "Q_MIN_VIOLATION"} for event in attempt.q_limit_events)

    def _can_attempt(self, attempt) -> bool:
        checker = getattr(attempt, "can_run", None)
        return bool(checker is None or checker())

    def _output_dir(self, case_file: Path, output_dir: str | Path | None) -> Path:
        if output_dir:
            return Path(output_dir).resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path.cwd() / "runs" / f"{case_file.stem}_{stamp}"
