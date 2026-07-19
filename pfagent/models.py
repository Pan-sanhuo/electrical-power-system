from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    location: str = ""
    suggestion: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
        }


@dataclass(slots=True)
class RepairAction:
    action: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "details": self.details}


@dataclass(slots=True)
class SolveAttempt:
    name: str
    engine: str
    options: dict[str, Any]
    success: bool
    elapsed_s: float
    result: dict[str, Any] | None = None
    error: str | None = None
    violations: list[dict[str, Any]] = field(default_factory=list)
    q_limit_events: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        blocking_q = {"Q_MAX_VIOLATION", "Q_MIN_VIOLATION"}
        return bool(
            self.success
            and not self.violations
            and not any(event.get("type") in blocking_q for event in self.q_limit_events)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine": self.engine,
            "options": self.options,
            "success": self.success,
            "feasible": self.feasible,
            "elapsed_s": round(self.elapsed_s, 4),
            "error": self.error,
            "violations": self.violations,
            "q_limit_events": self.q_limit_events,
            "diagnostics": self.diagnostics,
        }


@dataclass(slots=True)
class AgentRunReport:
    case_path: Path
    output_dir: Path
    validation: list[ValidationIssue]
    attempts: list[SolveAttempt]
    repairs: list[RepairAction]
    llm_sections: dict[str, Any]
    final_case_path: Path | None = None
    final_report_path: Path | None = None
    final_json_path: Path | None = None

    @property
    def success(self) -> bool:
        return bool(self.attempts and self.attempts[-1].feasible)

    @property
    def solver_converged(self) -> bool:
        return bool(self.attempts and self.attempts[-1].success)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_path": str(self.case_path),
            "output_dir": str(self.output_dir),
            "success": self.success,
            "validation": [issue.as_dict() for issue in self.validation],
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "repairs": [repair.as_dict() for repair in self.repairs],
            "llm_sections": self.llm_sections,
            "final_case_path": str(self.final_case_path) if self.final_case_path else None,
            "final_report_path": str(self.final_report_path) if self.final_report_path else None,
            "final_json_path": str(self.final_json_path) if self.final_json_path else None,
        }
