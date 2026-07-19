from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from pypower.loadcase import loadcase


REQUIRED_KEYS = ("baseMVA", "bus", "gen", "branch")


class CaseLoadError(RuntimeError):
    pass


def load_power_case(path: str | Path) -> dict[str, Any]:
    """Load PYPOWER/MATPOWER style data into a normalized ppc dict.

    Supported directly:
    - PYPOWER .py case files
    - MATPOWER/PYPOWER .mat files through pypower.loadcase
    - simple MATPOWER .m files with mpc.baseMVA/mpc.bus/mpc.gen/mpc.branch
    - JSON files storing a ppc/mpc dict
    """

    case_path = Path(path).resolve()
    if not case_path.exists():
        raise CaseLoadError(f"算例文件不存在: {case_path}")

    suffix = case_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(case_path.read_text(encoding="utf-8"))
        ppc = data.get("ppc") or data.get("mpc") or data
    elif suffix == ".m":
        ppc = parse_matpower_m_file(case_path)
    elif suffix == ".py":
        ppc = import_python_case_function(case_path)
    elif suffix == ".mat":
        ppc = loadcase(str(case_path))
        if isinstance(ppc, int):
            raise CaseLoadError(f"PYPOWER loadcase 读取失败，错误码: {ppc}")
    else:
        raise CaseLoadError(f"暂不支持的算例文件格式: {suffix}")

    return normalize_case(ppc)


def normalize_case(ppc: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_KEYS if key not in ppc]
    if missing:
        raise CaseLoadError(f"算例缺少字段: {', '.join(missing)}")

    normalized = copy.deepcopy(ppc)
    normalized["baseMVA"] = float(np.asarray(normalized["baseMVA"]).squeeze())
    normalized["version"] = str(normalized.get("version", "2"))
    for key in ("bus", "gen", "branch", "gencost", "areas"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = np.asarray(normalized[key], dtype=float)
    return normalized


def case_function_name(path: str | Path) -> str:
    stem = Path(path).stem
    return re.sub(r"\W+", "_", stem)


def parse_matpower_m_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = _strip_matlab_comments(text)
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9eE+\-.]+)\s*;", text)
    if not base_match:
        raise CaseLoadError("MATPOWER .m 文件中未找到 mpc.baseMVA")

    ppc: dict[str, Any] = {
        "version": "2",
        "baseMVA": float(base_match.group(1)),
    }
    for key in ("bus", "gen", "branch", "gencost", "areas"):
        match = re.search(rf"mpc\.{key}\s*=\s*\[(.*?)\]\s*;", text, flags=re.S)
        if match:
            ppc[key] = _parse_matlab_matrix(match.group(1), key)
    missing = [key for key in REQUIRED_KEYS if key not in ppc]
    if missing:
        raise CaseLoadError(f"MATPOWER .m 文件缺少矩阵: {', '.join(missing)}")
    return ppc


def _strip_matlab_comments(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        if "%" in line:
            line = line.split("%", 1)[0]
        cleaned.append(line.replace("...", " "))
    return "\n".join(cleaned)


def _parse_matlab_matrix(body: str, name: str) -> np.ndarray:
    rows = []
    for raw_row in body.replace("\r", "\n").split(";"):
        row = raw_row.strip()
        if not row:
            continue
        values = np.fromstring(row.replace(",", " "), sep=" ")
        if values.size:
            rows.append(values)
    if not rows:
        return np.empty((0, 0), dtype=float)

    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise CaseLoadError(f"MATPOWER 矩阵 {name} 的列数不一致: {sorted(widths)}")
    return np.vstack(rows).astype(float)


def import_python_case_function(path: str | Path) -> dict[str, Any]:
    case_path = Path(path).resolve()
    func_name = case_function_name(case_path)
    spec = importlib.util.spec_from_file_location(func_name, case_path)
    if spec is None or spec.loader is None:
        raise CaseLoadError(f"无法导入 PYPOWER 算例: {case_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, func_name):
        raise CaseLoadError(f"算例文件中未找到函数 {func_name}()")
    return normalize_case(getattr(module, func_name)())


def clone_case(ppc: dict[str, Any]) -> dict[str, Any]:
    return normalize_case(copy.deepcopy(ppc))


def write_pypower_case(ppc: dict[str, Any], path: str | Path, function_name: str | None = None) -> Path:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    func = function_name or case_function_name(out)

    lines = [
        "from numpy import array",
        "",
        "",
        f"def {func}():",
        "    ppc = {}",
        "    ppc['version'] = '2'",
        f"    ppc['baseMVA'] = {float(ppc['baseMVA']):.12g}",
    ]
    for key in ("bus", "gen", "branch", "gencost", "areas"):
        if key in ppc and ppc[key] is not None:
            lines.append(f"    ppc['{key}'] = array({_matrix_literal(np.asarray(ppc[key], dtype=float))})")
    lines.append("    return ppc")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_matpower_case(ppc: dict[str, Any], path: str | Path, function_name: str | None = None) -> Path:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    func = function_name or case_function_name(out)

    lines = [
        f"function mpc = {func}",
        "mpc.version = '2';",
        f"mpc.baseMVA = {float(ppc['baseMVA']):.12g};",
    ]
    for key in ("bus", "gen", "branch", "gencost", "areas"):
        if key in ppc and ppc[key] is not None:
            lines.append(f"mpc.{key} = [")
            for row in np.asarray(ppc[key], dtype=float):
                lines.append("    " + " ".join(f"{float(v):.12g}" for v in row) + ";")
            lines.append("];")
    lines.append("end")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_json_report(data: dict[str, Any], path: str | Path) -> Path:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _matrix_literal(matrix: np.ndarray) -> str:
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    rows = []
    for row in matrix:
        rows.append("[" + ", ".join(f"{float(v):.12g}" for v in row) + "]")
    return "[\n        " + ",\n        ".join(rows) + "\n    ]"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value
