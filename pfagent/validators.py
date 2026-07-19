from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
from pypower.idx_brch import (
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    T_BUS,
)
from pypower.idx_bus import BASE_KV, BUS_I, BUS_TYPE, NONE, PD, PQ, PV, QD, REF, VA, VM, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN, QG, QMAX, QMIN, VG

from .models import ValidationIssue


def validate_case(ppc: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    base = float(ppc.get("baseMVA", 0))
    bus = np.asarray(ppc.get("bus", []), dtype=float)
    gen = np.asarray(ppc.get("gen", []), dtype=float)
    branch = np.asarray(ppc.get("branch", []), dtype=float)

    if not np.isfinite(base) or base <= 0:
        issues.append(_issue("error", "BASE_MVA", "baseMVA 必须为正数", "baseMVA", "修正系统基准容量"))

    _check_shape(issues, "bus", bus, 13)
    _check_shape(issues, "gen", gen, 10)
    _check_shape(issues, "branch", branch, 13)
    if any(issue.level == "error" and issue.code == "SHAPE" for issue in issues):
        return issues

    for name, matrix in (("bus", bus), ("gen", gen), ("branch", branch)):
        if not np.all(np.isfinite(matrix)):
            issues.append(_issue("error", "NAN_INF", f"{name} 矩阵存在 NaN 或 Inf", name, "删除或修正异常数值"))

    bus_ids = bus[:, BUS_I].astype(int)
    if len(set(bus_ids.tolist())) != len(bus_ids):
        issues.append(_issue("error", "DUP_BUS", "母线编号存在重复", "bus[:, BUS_I]", "保证每个母线编号唯一"))

    bus_set = set(bus_ids.tolist())
    active_bus = bus[:, BUS_TYPE] != NONE
    ref_count = int(np.sum(bus[:, BUS_TYPE] == REF))
    if ref_count == 0:
        issues.append(_issue("error", "NO_REF", "系统缺少平衡节点 REF/slack", "bus[:, BUS_TYPE]", "选择一个电压支撑强的电源节点作为 REF"))
    elif ref_count > 1:
        issues.append(_issue("warning", "MULTI_REF", f"系统存在 {ref_count} 个 REF 节点", "bus[:, BUS_TYPE]", "确认是否为多岛系统；单岛系统通常只保留一个 REF"))

    illegal_types = sorted(set(bus[:, BUS_TYPE].astype(int).tolist()) - {PQ, PV, REF, NONE})
    if illegal_types:
        issues.append(_issue("error", "BUS_TYPE", f"存在非法母线类型: {illegal_types}", "bus[:, BUS_TYPE]", "使用 1=PQ, 2=PV, 3=REF, 4=NONE"))

    _check_bus_limits(issues, bus)
    _check_generators(issues, gen, bus, bus_set)
    _check_branches(issues, branch, bus_set)
    _check_islands(issues, bus, branch)
    _check_power_balance(issues, bus, gen)
    return issues


def summarize_case(ppc: dict[str, Any]) -> dict[str, Any]:
    bus = np.asarray(ppc["bus"], dtype=float)
    gen = np.asarray(ppc["gen"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)
    online_gen = gen[:, GEN_STATUS] > 0 if gen.shape[1] > GEN_STATUS else np.ones(gen.shape[0], dtype=bool)
    active_branch = branch[:, BR_STATUS] > 0 if branch.shape[1] > BR_STATUS else np.ones(branch.shape[0], dtype=bool)
    return {
        "baseMVA": float(ppc["baseMVA"]),
        "bus_count": int(bus.shape[0]),
        "gen_count": int(gen.shape[0]),
        "online_gen_count": int(np.sum(online_gen)),
        "branch_count": int(branch.shape[0]),
        "active_branch_count": int(np.sum(active_branch)),
        "total_load_p_mw": round(float(np.sum(bus[:, PD])), 6),
        "total_load_q_mvar": round(float(np.sum(bus[:, QD])), 6),
        "online_pg_mw": round(float(np.sum(gen[online_gen, PG])), 6),
        "ref_buses": bus[bus[:, BUS_TYPE] == REF, BUS_I].astype(int).tolist(),
        "pv_buses": bus[bus[:, BUS_TYPE] == PV, BUS_I].astype(int).tolist(),
        "pq_buses": bus[bus[:, BUS_TYPE] == PQ, BUS_I].astype(int).tolist(),
        "base_kv_values": sorted(set(np.round(bus[:, BASE_KV], 4).tolist())) if bus.shape[1] > BASE_KV else [],
    }


def compact_case_evidence(ppc: dict[str, Any], max_rows: int = 50) -> dict[str, Any]:
    """Provide bounded, structured source data for LLM review."""
    payload: dict[str, Any] = {
        "columns": {
            "bus": ["BUS_I", "BUS_TYPE", "PD", "QD", "GS", "BS", "AREA", "VM", "VA", "BASE_KV", "ZONE", "VMAX", "VMIN"],
            "gen": ["GEN_BUS", "PG", "QG", "QMAX", "QMIN", "VG", "MBASE", "GEN_STATUS", "PMAX", "PMIN"],
            "branch": ["F_BUS", "T_BUS", "BR_R", "BR_X", "BR_B", "RATE_A", "RATE_B", "RATE_C", "TAP", "SHIFT", "BR_STATUS"],
        },
        "truncated": {},
    }
    for key in ("bus", "gen", "branch"):
        matrix = np.asarray(ppc[key], dtype=float)
        payload[key] = matrix[:max_rows].tolist()
        payload["truncated"][key] = bool(matrix.shape[0] > max_rows)
    return payload


def q_limit_violations(input_ppc: dict[str, Any], result_ppc: dict[str, Any], tol: float = 1e-5) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    in_bus = np.asarray(input_ppc["bus"], dtype=float)
    out_bus = np.asarray(result_ppc["bus"], dtype=float)
    out_gen = np.asarray(result_ppc["gen"], dtype=float)
    in_types = {int(row[BUS_I]): int(row[BUS_TYPE]) for row in in_bus}
    out_types = {int(row[BUS_I]): int(row[BUS_TYPE]) for row in out_bus}

    for idx, row in enumerate(out_gen):
        if row.shape[0] <= max(QG, QMAX, QMIN, GEN_STATUS):
            continue
        if row[GEN_STATUS] <= 0:
            continue
        bus_id = int(row[GEN_BUS])
        before = in_types.get(bus_id)
        after = out_types.get(bus_id)
        qg, qmax, qmin = float(row[QG]), float(row[QMAX]), float(row[QMIN])
        if qg > qmax + tol:
            events.append(
                {
                    "type": "Q_MAX_VIOLATION",
                    "gen_index": idx,
                    "bus": bus_id,
                    "qg": round(qg, 6),
                    "limit": round(qmax, 6),
                    "message": "发电机无功超过上限，PV/REF 节点应按工程规则转为 PQ 或启用 enforce_q_lims",
                }
            )
        elif qg < qmin - tol:
            events.append(
                {
                    "type": "Q_MIN_VIOLATION",
                    "gen_index": idx,
                    "bus": bus_id,
                    "qg": round(qg, 6),
                    "limit": round(qmin, 6),
                    "message": "发电机无功低于下限，PV/REF 节点应按工程规则转为 PQ 或启用 enforce_q_lims",
                }
            )
        elif before in (PV, REF) and after == PQ and (abs(qg - qmax) <= 1e-4 or abs(qg - qmin) <= 1e-4):
            events.append(
                {
                    "type": "PV_TO_PQ",
                    "gen_index": idx,
                    "bus": bus_id,
                    "qg": round(qg, 6),
                    "limit": round(qmax if abs(qg - qmax) <= abs(qg - qmin) else qmin, 6),
                    "message": "无功达到限值，求解器已将 PV/REF 节点按无功限额处理为 PQ",
                }
            )
    return events


def operating_violations(result_ppc: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    bus = np.asarray(result_ppc["bus"], dtype=float)
    branch = np.asarray(result_ppc["branch"], dtype=float)

    for row in bus:
        if row[BUS_TYPE] == NONE:
            continue
        vm = float(row[VM])
        if vm < row[VMIN]:
            violations.append(
                {
                    "type": "LOW_VOLTAGE",
                    "target": int(row[BUS_I]),
                    "value": round(vm, 6),
                    "limit": round(float(row[VMIN]), 6),
                    "message": "母线电压低于下限",
                }
            )
        elif vm > row[VMAX]:
            violations.append(
                {
                    "type": "HIGH_VOLTAGE",
                    "target": int(row[BUS_I]),
                    "value": round(vm, 6),
                    "limit": round(float(row[VMAX]), 6),
                    "message": "母线电压高于上限",
                }
            )

    if branch.shape[1] > QT:
        for idx, row in enumerate(branch):
            if row[BR_STATUS] <= 0 or row[RATE_A] <= 0:
                continue
            s_from = float(np.hypot(row[PF], row[QF]))
            s_to = float(np.hypot(row[PT], row[QT]))
            loading = max(s_from, s_to) / float(row[RATE_A]) * 100
            if loading > 100:
                violations.append(
                    {
                        "type": "BRANCH_OVERLOAD",
                        "target": f"{int(row[F_BUS])}-{int(row[T_BUS])}",
                        "branch_index": idx,
                        "value": round(loading, 3),
                        "limit": 100.0,
                        "message": "线路或变压器潮流超过 RATE_A",
                    }
                )
    return violations


def summarize_result(result_ppc: dict[str, Any]) -> dict[str, Any]:
    bus = np.asarray(result_ppc["bus"], dtype=float)
    gen = np.asarray(result_ppc["gen"], dtype=float)
    branch = np.asarray(result_ppc["branch"], dtype=float)
    summary: dict[str, Any] = {
        "voltage_min": round(float(np.min(bus[:, VM])), 6),
        "voltage_max": round(float(np.max(bus[:, VM])), 6),
        "angle_min_deg": round(float(np.min(bus[:, VA])), 6),
        "angle_max_deg": round(float(np.max(bus[:, VA])), 6),
        "total_pg_mw": round(float(np.sum(gen[gen[:, GEN_STATUS] > 0, PG])), 6),
        "total_qg_mvar": round(float(np.sum(gen[gen[:, GEN_STATUS] > 0, QG])), 6),
    }
    if branch.shape[1] > QT:
        loadings = []
        for row in branch:
            if row[BR_STATUS] > 0 and row[RATE_A] > 0:
                loadings.append(max(float(np.hypot(row[PF], row[QF])), float(np.hypot(row[PT], row[QT]))) / float(row[RATE_A]) * 100)
        summary["max_branch_loading_pct"] = round(float(max(loadings)), 6) if loadings else None
    return summary


def has_blocking_validation_errors(issues: list[ValidationIssue]) -> bool:
    blocking = {"BASE_MVA", "SHAPE", "NAN_INF", "DUP_BUS", "NO_REF", "BUS_TYPE", "GEN_BUS", "BRANCH_BUS", "ISLAND_NO_REF"}
    return any(issue.level == "error" and issue.code in blocking for issue in issues)


def _check_shape(issues: list[ValidationIssue], name: str, matrix: np.ndarray, min_cols: int) -> None:
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < min_cols:
        issues.append(_issue("error", "SHAPE", f"{name} 矩阵形状不正确，至少需要 {min_cols} 列", name, "检查是否为 MATPOWER/PYPOWER v2 格式"))


def _check_bus_limits(issues: list[ValidationIssue], bus: np.ndarray) -> None:
    for idx, row in enumerate(bus):
        loc = f"bus[{idx}]"
        if row[VMIN] >= row[VMAX]:
            issues.append(_issue("error", "V_LIMIT", "母线电压下限不小于上限", loc, "修正 VMIN/VMAX"))
        if not (0.2 <= row[VM] <= 2.0):
            issues.append(_issue("warning", "VM_INIT", f"母线初值 VM={row[VM]:.4g} 异常", loc, "使用 1.0 或接近额定电压的初值"))
        if abs(row[VA]) > 180:
            issues.append(_issue("warning", "VA_INIT", f"母线相角初值 VA={row[VA]:.4g} deg 异常", loc, "检查相角单位是否误填为弧度或数值漂移"))


def _check_generators(issues: list[ValidationIssue], gen: np.ndarray, bus: np.ndarray, bus_set: set[int]) -> None:
    bus_type = {int(row[BUS_I]): int(row[BUS_TYPE]) for row in bus}
    online_by_bus: dict[int, int] = defaultdict(int)
    for idx, row in enumerate(gen):
        loc = f"gen[{idx}]"
        bus_id = int(row[GEN_BUS])
        if bus_id not in bus_set:
            issues.append(_issue("error", "GEN_BUS", f"发电机接入不存在的母线 {bus_id}", loc, "修正 GEN_BUS 或补齐母线"))
            continue
        if row[GEN_STATUS] <= 0:
            continue
        online_by_bus[bus_id] += 1
        if row[QMAX] < row[QMIN]:
            issues.append(_issue("error", "Q_LIMIT", "发电机 QMAX 小于 QMIN", loc, "修正无功上下限"))
        if row[PMAX] < row[PMIN]:
            issues.append(_issue("error", "P_LIMIT", "发电机 PMAX 小于 PMIN", loc, "修正有功上下限"))
        if bus_type.get(bus_id) != REF and (row[PG] > row[PMAX] + 1e-6 or row[PG] < row[PMIN] - 1e-6):
            issues.append(_issue("warning", "PG_LIMIT", "发电机 PG 初值超出 PMIN/PMAX", loc, "调整机组出力初值或出力上下限"))
        if not (0.85 <= row[VG] <= 1.15):
            issues.append(_issue("warning", "VG_SETPOINT", f"发电机电压给定 VG={row[VG]:.4g} 可疑", loc, "确认电压给定是否为标幺值"))
    for idx, row in enumerate(bus):
        if row[BUS_TYPE] in (PV, REF) and online_by_bus[int(row[BUS_I])] == 0:
            issues.append(_issue("warning", "PV_WITHOUT_GEN", "PV/REF 母线没有在线发电机", f"bus[{idx}]", "将其改为 PQ 或补齐在线电源"))
        if row[BUS_TYPE] == PQ and online_by_bus[int(row[BUS_I])] > 0:
            issues.append(_issue("info", "GEN_ON_PQ", "PQ 母线上存在在线发电机，求解时不会控制电压", f"bus[{idx}]", "若需控压，应改为 PV 或 REF"))


def _check_branches(issues: list[ValidationIssue], branch: np.ndarray, bus_set: set[int]) -> None:
    for idx, row in enumerate(branch):
        loc = f"branch[{idx}]"
        f_bus, t_bus = int(row[F_BUS]), int(row[T_BUS])
        if f_bus not in bus_set or t_bus not in bus_set:
            issues.append(_issue("error", "BRANCH_BUS", f"支路端点 {f_bus}-{t_bus} 引用了不存在的母线", loc, "修正 F_BUS/T_BUS"))
        if row[BR_STATUS] <= 0:
            continue
        if abs(row[BR_R]) < 1e-12 and abs(row[BR_X]) < 1e-12:
            issues.append(_issue("error", "ZERO_IMPEDANCE", f"支路 {f_bus}-{t_bus} 阻抗为零", loc, "填入合理阻抗或用母线合并建模"))
        elif abs(row[BR_X]) < 1e-8:
            issues.append(_issue("warning", "LOW_REACTANCE", f"支路 {f_bus}-{t_bus} 电抗接近零，Jacobi 可能病态", loc, "确认变压器或等值支路参数"))
        if row[RATE_A] < 0:
            issues.append(_issue("warning", "RATE_NEGATIVE", "RATE_A 为负数", loc, "修正热稳定限额；若无限额可填 0"))


def _check_islands(issues: list[ValidationIssue], bus: np.ndarray, branch: np.ndarray) -> None:
    active_buses = {int(row[BUS_I]) for row in bus if row[BUS_TYPE] != NONE}
    if not active_buses:
        issues.append(_issue("error", "NO_ACTIVE_BUS", "没有可计算的在线母线", "bus", "至少保留一个 PQ/PV/REF 母线"))
        return
    graph: dict[int, set[int]] = {bus_id: set() for bus_id in active_buses}
    for row in branch:
        if row[BR_STATUS] <= 0:
            continue
        f_bus, t_bus = int(row[F_BUS]), int(row[T_BUS])
        if f_bus in graph and t_bus in graph:
            graph[f_bus].add(t_bus)
            graph[t_bus].add(f_bus)

    seen: set[int] = set()
    components: list[set[int]] = []
    for bus_id in sorted(active_buses):
        if bus_id in seen:
            continue
        comp = set()
        queue: deque[int] = deque([bus_id])
        seen.add(bus_id)
        while queue:
            node = queue.popleft()
            comp.add(node)
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(comp)

    ref_buses = {int(row[BUS_I]) for row in bus if row[BUS_TYPE] == REF}
    if len(components) > 1:
        sizes = [len(comp) for comp in components]
        issues.append(_issue("error", "ISLAND", f"网络存在 {len(components)} 个电气岛，规模 {sizes}", "branch", "每个可计算岛需要平衡节点，或合并/切除孤岛"))
    for comp in components:
        if not comp & ref_buses:
            issues.append(_issue("error", "ISLAND_NO_REF", f"电气岛 {sorted(comp)} 缺少 REF 节点", "bus/branch", "为该岛设置平衡节点或将孤岛退出计算"))


def _check_power_balance(issues: list[ValidationIssue], bus: np.ndarray, gen: np.ndarray) -> None:
    online = gen[:, GEN_STATUS] > 0
    total_load = float(np.sum(bus[:, PD]))
    total_pg = float(np.sum(gen[online, PG])) if gen.size else 0.0
    total_pmax = float(np.sum(gen[online, PMAX])) if gen.size else 0.0
    if total_pmax + 1e-6 < total_load:
        issues.append(_issue("warning", "P_CAPACITY", f"在线机组 PMAX={total_pmax:.3f} MW 小于总负荷 {total_load:.3f} MW", "gen/bus", "增加在线机组、降低负荷或重新分配出力"))
    if total_pg <= 0 and total_load > 0:
        issues.append(_issue("warning", "NO_INITIAL_PG", "有负荷但在线机组 PG 初值非正", "gen[:, PG]", "提供合理初始出力，避免平衡机承担过大不平衡"))


def _issue(level: str, code: str, message: str, location: str = "", suggestion: str = "") -> ValidationIssue:
    return ValidationIssue(level=level, code=code, message=message, location=location, suggestion=suggestion)
