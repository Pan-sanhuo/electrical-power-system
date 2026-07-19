from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import hstack, vstack
from pypower.bustypes import bustypes
from pypower.dSbus_dV import dSbus_dV
from pypower.ext2int import ext2int
from pypower.idx_bus import VA, VM
from pypower.makeYbus import makeYbus
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, PQ, PV, QD, REF
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, QG, QMAX, QMIN
from pypower.api import runpf
from pypower.ppoption import ppoption

from .caseio import clone_case, normalize_case, write_matpower_case
from .validators import q_limit_violations


@dataclass(slots=True)
class PowerFlowOptions:
    engine: str = "pypower"
    pf_alg: int = 1
    enforce_q_lims: bool = True
    tol: float = 1e-8
    max_it: int = 20
    verbose: int = 0
    out_all: int = 0
    matpower_path: str | None = None
    matlab_command: str = "matlab"

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "pf_alg": self.pf_alg,
            "pf_alg_name": algorithm_name(self.pf_alg),
            "enforce_q_lims": self.enforce_q_lims,
            "tol": self.tol,
            "max_it": self.max_it,
            "verbose": self.verbose,
            "out_all": self.out_all,
            "matpower_path": self.matpower_path,
            "matlab_command": self.matlab_command,
        }


@dataclass(slots=True)
class SolverResult:
    success: bool
    ppc: dict[str, Any] | None
    elapsed_s: float
    error: str | None = None
    raw: dict[str, Any] | None = None


def algorithm_name(pf_alg: int) -> str:
    return {
        1: "Newton-Raphson",
        2: "Fast-Decoupled XB",
        3: "Fast-Decoupled BX",
        4: "Gauss-Seidel",
    }.get(int(pf_alg), f"PF_ALG={pf_alg}")


class PyPowerSolver:
    engine = "pypower"

    def run(self, ppc: dict[str, Any], options: PowerFlowOptions) -> SolverResult:
        if options.enforce_q_lims:
            return self._run_with_manual_q_limits(ppc, options)
        return self._run_once(ppc, options, enforce_q_lims=False)

    def _run_once(self, ppc: dict[str, Any], options: PowerFlowOptions, enforce_q_lims: bool) -> SolverResult:
        case = clone_case(ppc)
        diagnostics = numerical_diagnostics(case)
        start = time.perf_counter()
        try:
            ppopt = ppoption(
                VERBOSE=options.verbose,
                OUT_ALL=options.out_all,
                PF_ALG=int(options.pf_alg),
                PF_TOL=float(options.tol),
                PF_MAX_IT=int(options.max_it),
                PF_MAX_IT_FD=max(30, int(options.max_it) * 3),
                PF_MAX_IT_GS=max(1000, int(options.max_it) * 50),
                ENFORCE_Q_LIMS=bool(enforce_q_lims),
            )
            result, success = runpf(case, ppopt)
            elapsed = time.perf_counter() - start
            return SolverResult(bool(success), normalize_case(result), elapsed, raw={"diagnostics": diagnostics})
        except Exception as exc:  # PYPOWER raises varied linear algebra/index errors.
            elapsed = time.perf_counter() - start
            return SolverResult(
                False,
                None,
                elapsed,
                error=f"{type(exc).__name__}: {exc}",
                raw={"diagnostics": diagnostics, "failure_class": classify_solver_error(exc)},
            )

    def _run_with_manual_q_limits(self, ppc: dict[str, Any], options: PowerFlowOptions) -> SolverResult:
        """Emulate MATPOWER/PYPOWER Q-limit enforcement without PYPOWER's NumPy-2 indexing bug."""

        start = time.perf_counter()
        current = clone_case(ppc)
        limited: list[dict[str, Any]] = []
        total_elapsed = 0.0
        last_result: SolverResult | None = None

        for _ in range(max(1, current["gen"].shape[0] + 1)):
            result = self._run_once(current, options, enforce_q_lims=False)
            total_elapsed += result.elapsed_s
            last_result = result
            if not result.success or result.ppc is None:
                result.elapsed_s = time.perf_counter() - start
                return result

            events = [event for event in q_limit_violations(current, result.ppc) if event["type"] in {"Q_MAX_VIOLATION", "Q_MIN_VIOLATION"}]
            if not events:
                restored = _restore_limited_generators(result.ppc, limited)
                raw = dict(result.raw or {})
                raw["manual_q_limits"] = limited
                return SolverResult(True, restored, time.perf_counter() - start, raw=raw)

            for event in events:
                gen_idx = int(event["gen_index"])
                if any(item["gen_index"] == gen_idx for item in limited):
                    continue
                limit = float(event["limit"])
                record = _limit_generator_as_fixed_injection(current, gen_idx, limit)
                limited.append(record)
            if not _ensure_reference_bus(current):
                return SolverResult(
                    False,
                    None,
                    time.perf_counter() - start,
                    error="启用无功限额后没有可用 REF/PV 电源承担平衡，当前运行方式无功支撑不足",
                    raw={
                        "manual_q_limits": limited,
                        "failure_class": "reactive_power_infeasible",
                        "diagnostics": (result.raw or {}).get("diagnostics", {}),
                    },
                )

        if last_result and last_result.ppc is not None:
            restored = _restore_limited_generators(last_result.ppc, limited)
            return SolverResult(False, restored, time.perf_counter() - start, error="无功限额循环达到上限，仍存在越限", raw={"manual_q_limits": limited})
        return SolverResult(False, None, total_elapsed, error="无功限额循环未获得结果", raw={"manual_q_limits": limited})


def classify_solver_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(token in text for token in ("singular", "exactly singular", "factor is exactly singular")):
        return "singular_jacobian"
    if any(token in text for token in ("nan", "inf", "overflow")):
        return "invalid_numerical_value"
    if "index" in text:
        return "data_or_library_index_error"
    return "solver_exception"


def numerical_diagnostics(ppc: dict[str, Any]) -> dict[str, Any]:
    """Estimate the initial Newton Jacobian quality for explainable diagnosis."""
    try:
        internal = ext2int(clone_case(ppc))
        base_mva = internal["baseMVA"]
        bus = internal["bus"]
        gen = internal["gen"]
        branch = internal["branch"]
        ref, pv, pq = bustypes(bus, gen)
        pvpq = np.r_[pv, pq].astype(int)
        pq = np.asarray(pq, dtype=int)
        voltage = bus[:, VM] * np.exp(1j * np.deg2rad(bus[:, VA]))
        ybus, _yf, _yt = makeYbus(base_mva, bus, branch)
        d_s_d_vm, d_s_d_va = dSbus_dV(ybus, voltage)
        j11 = d_s_d_va[np.ix_(pvpq, pvpq)].real
        j12 = d_s_d_vm[np.ix_(pvpq, pq)].real
        j21 = d_s_d_va[np.ix_(pq, pvpq)].imag
        j22 = d_s_d_vm[np.ix_(pq, pq)].imag
        jacobian = vstack([hstack([j11, j12]), hstack([j21, j22])], format="csr").toarray()
        if jacobian.size == 0:
            return {"available": False, "reason": "empty_jacobian"}
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        condition = float(np.linalg.cond(jacobian))
        rank = int(np.linalg.matrix_rank(jacobian))
        return {
            "available": True,
            "jacobian_size": list(jacobian.shape),
            "jacobian_rank": rank,
            "jacobian_condition": condition if np.isfinite(condition) else None,
            "jacobian_min_singular_value": float(np.min(singular_values)),
            "numerically_singular": bool(rank < min(jacobian.shape) or not np.isfinite(condition) or condition > 1e12),
            "ref_count": int(len(ref)),
            "pv_count": int(len(pv)),
            "pq_count": int(len(pq)),
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _limit_generator_as_fixed_injection(ppc: dict[str, Any], gen_idx: int, q_limit: float) -> dict[str, Any]:
    gen = ppc["gen"]
    bus = ppc["bus"]
    row = gen[gen_idx]
    bus_id = int(row[GEN_BUS])
    bus_rows = np.where(bus[:, BUS_I].astype(int) == bus_id)[0]
    if not bus_rows.size:
        raise ValueError(f"发电机 {gen_idx} 接入不存在的母线 {bus_id}")
    bus_idx = int(bus_rows[0])
    record = {
        "gen_index": gen_idx,
        "bus": bus_id,
        "bus_index": bus_idx,
        "pg": float(row[PG]),
        "qg": float(q_limit),
        "old_bus_type": int(bus[bus_idx, BUS_TYPE]),
    }
    row[QG] = q_limit
    row[GEN_STATUS] = 0
    bus[bus_idx, PD] -= row[PG]
    bus[bus_idx, QD] -= row[QG]
    other_online = np.any(
        (gen[:, GEN_BUS].astype(int) == bus_id)
        & (gen[:, GEN_STATUS] > 0)
    )
    if not other_online:
        bus[bus_idx, BUS_TYPE] = PQ
    return record


def _restore_limited_generators(result_ppc: dict[str, Any], limited: list[dict[str, Any]]) -> dict[str, Any]:
    restored = normalize_case(result_ppc)
    bus = restored["bus"]
    gen = restored["gen"]
    limited_indices = {int(item["gen_index"]) for item in limited}
    for item in limited:
        gen_idx = int(item["gen_index"])
        bus_id = int(item["bus"])
        bus_rows = np.where(bus[:, BUS_I].astype(int) == bus_id)[0]
        if not bus_rows.size:
            continue
        bus_idx = int(bus_rows[0])
        gen[gen_idx, GEN_STATUS] = 1
        gen[gen_idx, PG] = float(item["pg"])
        gen[gen_idx, QG] = float(item["qg"])
        bus[bus_idx, PD] += float(item["pg"])
        bus[bus_idx, QD] += float(item["qg"])
        other_controlling_gen = any(
            idx not in limited_indices
            and int(row[GEN_BUS]) == bus_id
            and row[GEN_STATUS] > 0
            for idx, row in enumerate(gen)
        )
        if not other_controlling_gen:
            bus[bus_idx, BUS_TYPE] = PQ
    return restored


def _ensure_reference_bus(ppc: dict[str, Any]) -> bool:
    bus = ppc["bus"]
    gen = ppc["gen"]
    if np.any(bus[:, BUS_TYPE] == REF):
        return True
    candidates = []
    for row in gen:
        if row[GEN_STATUS] <= 0:
            continue
        bus_id = int(row[GEN_BUS])
        bus_idx = np.where(bus[:, BUS_I].astype(int) == bus_id)[0]
        if bus_idx.size:
            candidates.append((float(row[PMAX]), int(bus_idx[0])))
    if candidates:
        _, idx = max(candidates, key=lambda item: item[0])
        bus[idx, BUS_TYPE] = REF
        return True
    else:
        pv_rows = np.where(bus[:, BUS_TYPE] == PV)[0]
        if pv_rows.size:
            bus[int(pv_rows[0]), BUS_TYPE] = REF
            return True
    return False


class MatpowerSolver:
    engine = "matpower"

    def run(self, ppc: dict[str, Any], options: PowerFlowOptions, work_dir: str | Path | None = None) -> SolverResult:
        matlab = shutil.which(options.matlab_command)
        if matlab is None:
            return SolverResult(False, None, 0.0, error=f"未找到 MATLAB 命令: {options.matlab_command}")
        if not options.matpower_path:
            return SolverResult(False, None, 0.0, error="未配置 MATPOWER 路径，请传入 --matpower-path 或设置 MATPOWER_PATH")
        matpower_path = Path(options.matpower_path).resolve()
        if not matpower_path.exists():
            return SolverResult(False, None, 0.0, error=f"MATPOWER 路径不存在: {matpower_path}")

        start = time.perf_counter()
        with tempfile.TemporaryDirectory(dir=str(work_dir) if work_dir else None) as tmp:
            tmp_path = Path(tmp)
            case_file = write_matpower_case(ppc, tmp_path / "agent_case.m", "agent_case")
            json_file = tmp_path / "matpower_result.json"
            script_file = tmp_path / "run_agent_pf.m"
            script_file.write_text(
                _matpower_script(matpower_path, case_file, json_file, options),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [matlab, "-batch", f"run('{_matlab_path(script_file)}')"],
                cwd=str(tmp_path),
                text=True,
                capture_output=True,
                timeout=300,
            )
            elapsed = time.perf_counter() - start
            if proc.returncode != 0:
                return SolverResult(False, None, elapsed, error=(proc.stderr or proc.stdout).strip())
            if not json_file.exists():
                return SolverResult(False, None, elapsed, error="MATPOWER 未生成结果 JSON")
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            if "error" in payload:
                return SolverResult(False, None, elapsed, error=payload["error"], raw=payload)
            ppc_result = {
                "version": "2",
                "baseMVA": payload["baseMVA"],
                "bus": np.asarray(payload["bus"], dtype=float),
                "gen": np.asarray(payload["gen"], dtype=float),
                "branch": np.asarray(payload["branch"], dtype=float),
            }
            if "gencost" in payload:
                ppc_result["gencost"] = np.asarray(payload["gencost"], dtype=float)
            return SolverResult(bool(payload.get("success", False)), normalize_case(ppc_result), elapsed, raw=payload)


def _matpower_script(matpower_path: Path, case_file: Path, json_file: Path, options: PowerFlowOptions) -> str:
    q_lims = "1" if options.enforce_q_lims else "0"
    alg = {1: "NR", 2: "FDXB", 3: "FDBX", 4: "GS"}.get(int(options.pf_alg), "NR")
    return f"""
try
    addpath(genpath('{_matlab_path(matpower_path)}'));
    mpopt = mpoption('verbose', {int(options.verbose)}, 'out.all', {int(options.out_all)}, ...
        'pf.alg', '{alg}', 'pf.enforce_q_lims', {q_lims}, ...
        'pf.tol', {float(options.tol):.12g}, 'pf.nr.max_it', {int(options.max_it)}, ...
        'pf.fd.max_it', {max(30, int(options.max_it) * 3)}, 'pf.gs.max_it', {max(1000, int(options.max_it) * 50)});
    [results, success] = runpf('{_matlab_path(case_file)}', mpopt);
    out = struct();
    out.success = logical(success);
    out.baseMVA = results.baseMVA;
    out.bus = results.bus;
    out.gen = results.gen;
    out.branch = results.branch;
    if isfield(results, 'gencost')
        out.gencost = results.gencost;
    end
    fid = fopen('{_matlab_path(json_file)}', 'w');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
catch ME
    out = struct();
    out.error = getReport(ME, 'extended', 'hyperlinks', 'off');
    fid = fopen('{_matlab_path(json_file)}', 'w');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
    rethrow(ME);
end
"""


def _matlab_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")
