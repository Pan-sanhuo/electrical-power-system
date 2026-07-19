from __future__ import annotations

from typing import Any

import numpy as np
from pypower.idx_brch import BR_R, BR_STATUS, BR_X, RATE_A
from pypower.idx_bus import BUS_I, BUS_TYPE, NONE, PD, PQ, PV, QD, REF, VA, VM, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN, QMAX, QMIN, VG

from .caseio import clone_case
from .models import RepairAction


def repair_obvious_data_errors(ppc: dict[str, Any]) -> tuple[dict[str, Any], list[RepairAction]]:
    case = clone_case(ppc)
    actions: list[RepairAction] = []
    bus = case["bus"]
    gen = case["gen"]
    branch = case["branch"]
    bus_types = {int(row[BUS_I]): int(row[BUS_TYPE]) for row in bus}

    if not np.any(bus[:, BUS_TYPE] == REF):
        ref_bus = _largest_online_generator_bus(gen) or int(bus[0, BUS_I])
        row = np.where(bus[:, BUS_I].astype(int) == ref_bus)[0]
        if row.size:
            bus[row[0], BUS_TYPE] = REF
            actions.append(RepairAction("set_reference_bus", "算例缺少平衡节点，选择在线容量最大的电源母线作为 REF", {"bus": ref_bus}))

    for idx, row in enumerate(bus):
        details = {"bus": int(row[BUS_I]), "row": idx}
        if row[VMIN] >= row[VMAX]:
            row[VMIN], row[VMAX] = 0.94, 1.06
            actions.append(RepairAction("fix_voltage_limits", "母线电压上下限顺序错误，恢复常用运行限额", details | {"vmin": 0.94, "vmax": 1.06}))
        if not (0.5 <= row[VM] <= 1.5):
            old = float(row[VM])
            row[VM] = 1.0
            actions.append(RepairAction("fix_voltage_initial_value", "母线电压初值明显异常，重置为 1.0 p.u.", details | {"old_vm": old, "new_vm": 1.0}))
        if abs(row[VA]) > 180:
            old = float(row[VA])
            row[VA] = 0.0
            actions.append(RepairAction("fix_angle_initial_value", "母线相角初值明显异常，重置为 0 deg", details | {"old_va": old, "new_va": 0.0}))

    for idx, row in enumerate(gen):
        details = {"gen_index": idx, "bus": int(row[GEN_BUS])}
        if row[QMAX] < row[QMIN]:
            old = (float(row[QMIN]), float(row[QMAX]))
            row[QMIN], row[QMAX] = row[QMAX], row[QMIN]
            actions.append(RepairAction("swap_q_limits", "发电机无功上下限颠倒，已交换 QMIN/QMAX", details | {"old": old}))
        if row[PMAX] < row[PMIN]:
            old = (float(row[PMIN]), float(row[PMAX]))
            row[PMIN], row[PMAX] = row[PMAX], row[PMIN]
            actions.append(RepairAction("swap_p_limits", "发电机有功上下限颠倒，已交换 PMIN/PMAX", details | {"old": old}))
        if row[GEN_STATUS] > 0:
            old_pg = float(row[PG])
            if bus_types.get(int(row[GEN_BUS])) != REF:
                row[PG] = min(max(row[PG], row[PMIN]), row[PMAX])
            if abs(row[PG] - old_pg) > 1e-9:
                actions.append(RepairAction("clip_pg_to_limits", "发电机 PG 初值超限，已夹到 PMIN/PMAX 内", details | {"old_pg": old_pg, "new_pg": float(row[PG])}))
            old_vg = float(row[VG])
            row[VG] = min(max(row[VG], 0.95), 1.08)
            if abs(row[VG] - old_vg) > 1e-9:
                actions.append(RepairAction("clip_vg_setpoint", "发电机电压给定明显异常，夹到 0.95-1.08 p.u.", details | {"old_vg": old_vg, "new_vg": float(row[VG])}))

    for idx, row in enumerate(branch):
        details = {"branch_index": idx}
        if row[RATE_A] < 0:
            old = float(row[RATE_A])
            row[RATE_A] = 0.0
            actions.append(RepairAction("fix_negative_rate", "支路 RATE_A 为负，改为 0 表示不检查热限额", details | {"old_rate_a": old}))
        if row[BR_STATUS] > 0 and abs(row[BR_R]) < 1e-12 and abs(row[BR_X]) < 1e-12:
            row[BR_X] = 1e-5
            actions.append(
                RepairAction(
                    "regularize_zero_impedance",
                    "支路零阻抗会导致导纳矩阵/Jacobi 病态，加入极小电抗作为数值应急；工程应用中应回填真实参数或合并母线",
                    details | {"new_x": 1e-5},
                )
            )
    return case, actions


def apply_flat_start(ppc: dict[str, Any]) -> tuple[dict[str, Any], RepairAction]:
    case = clone_case(ppc)
    bus = case["bus"]
    gen = case["gen"]
    bus[:, VA] = 0.0
    for row in bus:
        if row[BUS_TYPE] != NONE:
            row[VM] = 1.0
    for row in gen:
        if row[GEN_STATUS] <= 0:
            continue
        bus_idx = np.where(bus[:, BUS_I].astype(int) == int(row[GEN_BUS]))[0]
        if bus_idx.size:
            bus[bus_idx[0], VM] = min(max(row[VG], 0.95), 1.08)
    return case, RepairAction("flat_start", "牛顿初值可能远离可行解，改用平坦启动并保持 PV/REF 电压给定", {})


def redispatch_generation(ppc: dict[str, Any], reserve_factor: float = 1.02) -> tuple[dict[str, Any], RepairAction]:
    case = clone_case(ppc)
    bus = case["bus"]
    gen = case["gen"]
    online = gen[:, GEN_STATUS] > 0
    if not np.any(online):
        return case, RepairAction("redispatch_generation_skipped", "没有在线机组，无法重新分配有功出力", {})
    target = float(np.sum(bus[:, PD])) * reserve_factor
    pmin = gen[online, PMIN]
    pmax = gen[online, PMAX]
    headroom = np.maximum(pmax - pmin, 0.0)
    if float(np.sum(headroom)) <= 1e-9:
        return case, RepairAction("redispatch_generation_skipped", "在线机组没有可分配有功裕度", {"target_pg": target})
    dispatchable = max(target - float(np.sum(pmin)), 0.0)
    gen[online, PG] = pmin + headroom / float(np.sum(headroom)) * min(dispatchable, float(np.sum(headroom)))
    return case, RepairAction("redispatch_generation", "按机组有功裕度重新分配 PG，减轻平衡机承担的初始不平衡", {"target_pg": round(target, 6)})


def scale_load(ppc: dict[str, Any], factor: float, reason: str = "按比例削减负荷以恢复可行潮流") -> tuple[dict[str, Any], RepairAction]:
    case = clone_case(ppc)
    case["bus"][:, PD] *= factor
    case["bus"][:, QD] *= factor
    return case, RepairAction("scale_load", reason, {"factor": factor})


def tune_voltage_setpoints(ppc: dict[str, Any], direction: str) -> tuple[dict[str, Any], RepairAction]:
    case = clone_case(ppc)
    bus = case["bus"]
    gen = case["gen"]
    online = gen[:, GEN_STATUS] > 0
    if direction == "raise":
        gen[online, VG] = np.minimum(gen[online, VG] + 0.01, 1.06)
        for row in gen[online]:
            bus_idx = np.where(bus[:, BUS_I].astype(int) == int(row[GEN_BUS]))[0]
            if bus_idx.size and bus[bus_idx[0], BUS_TYPE] in (PV, REF):
                bus[bus_idx[0], VM] = row[VG]
        return case, RepairAction("raise_voltage_setpoints", "存在低电压风险，适当提高在线 PV/REF 电源电压给定", {"delta": 0.01, "cap": 1.06})
    gen[online, VG] = np.maximum(gen[online, VG] - 0.01, 0.98)
    for row in gen[online]:
        bus_idx = np.where(bus[:, BUS_I].astype(int) == int(row[GEN_BUS]))[0]
        if bus_idx.size and bus[bus_idx[0], BUS_TYPE] in (PV, REF):
            bus[bus_idx[0], VM] = row[VG]
    return case, RepairAction("lower_voltage_setpoints", "存在高电压风险，适当降低在线 PV/REF 电源电压给定", {"delta": -0.01, "floor": 0.98})


def _largest_online_generator_bus(gen: np.ndarray) -> int | None:
    if gen.size == 0:
        return None
    online = np.where(gen[:, GEN_STATUS] > 0)[0]
    if not online.size:
        return None
    best = online[np.argmax(gen[online, PMAX])]
    return int(gen[best, GEN_BUS])
