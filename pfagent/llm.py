from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LLMConfig:
    provider: str = "off"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    temperature: float = 0.2

    @property
    def enabled(self) -> bool:
        return self.provider.lower() not in {"", "off", "none", "rule", "rules"} and bool(self.api_key)

    @classmethod
    def from_env(
        cls,
        provider: str = "off",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> "LLMConfig":
        p = provider.lower()
        if p in {"off", "none", "rule", "rules"}:
            return cls(provider="off")
        if p == "deepseek":
            return cls(
                provider="deepseek",
                model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            )
        if p in {"kimi", "moonshot"}:
            return cls(
                provider="kimi",
                model=model or os.getenv("KIMI_MODEL") or os.getenv("MOONSHOT_MODEL", "moonshot-v1-32k"),
                api_key=api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"),
                base_url=base_url or os.getenv("KIMI_BASE_URL") or os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
            )
        return cls(
            provider=p,
            model=model or os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini"),
            api_key=api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
        )


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.enabled:
            return {
                "enabled": False,
                "provider": self.config.provider,
                "message": "未配置大模型 API key，已跳过 LLM 调用，使用规则诊断。",
            }

        endpoint = _chat_endpoint(self.config.base_url)
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            return {"enabled": True, "error": f"HTTP {exc.code}: {body[:1000]}", "provider": self.config.provider}
        except Exception as exc:
            return {"enabled": True, "error": f"{type(exc).__name__}: {exc}", "provider": self.config.provider}

        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _parse_jsonish(content)
        return {
            "enabled": True,
            "provider": self.config.provider,
            "model": self.config.model,
            "content": content,
            "parsed": parsed,
        }


def inspect_data_with_llm(client: LLMClient, payload: dict[str, Any]) -> dict[str, Any]:
    return client.chat_json(
        "你是资深电力系统潮流计算工程师。请检查 MATPOWER/PYPOWER 原始算例数据，"
        "识别拓扑、节点类型、机组限额、标幺值、孤岛、初值和工程合理性问题。"
        "每条判断必须引用输入中的母线号、机组序号或支路端点作为证据；不确定时明确说明。"
        "只输出 JSON，字段包括 risk_level, data_errors, engineering_warnings, suggested_repairs。",
        payload,
    )


def diagnose_result_with_llm(client: LLMClient, payload: dict[str, Any]) -> dict[str, Any]:
    return client.chat_json(
        "你是资深调度运行与潮流计算专家。请分析潮流计算过程和结果，尤其关注不收敛、"
        "Jacobi 病态或奇异、PV 节点无功越限转 PQ、电压越限、线路过载和不符合工程实际的运行方式。"
        "必须区分数值不收敛、数据错误和物理不可行；不得把仅数值收敛称为工程可行。"
        "只输出 JSON，字段包括 conclusion, root_causes, engineering_risks, recommended_actions。",
        payload,
    )


def propose_repairs_with_llm(client: LLMClient, payload: dict[str, Any]) -> dict[str, Any]:
    return client.chat_json(
        "你要为潮流不收敛或无解算例提出可执行修复动作。只能从 allowed_actions 中选择，"
        "不要编造字段。优先采用工程上可解释的措施：启用无功限额、改进初值、换算法、"
        "修正明显错误数据、重新分配机组出力、必要时负荷削减。根据 validation、diagnostics 和 attempts 排序，"
        "不要建议无法由 allowed_actions 执行的动作。只输出 JSON，字段包括 actions 和 rationale。",
        payload,
    )


def _chat_endpoint(base_url: str | None) -> str:
    if not base_url:
        raise ValueError("LLM base_url 未配置")
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return trimmed + "/chat/completions"


def _parse_jsonish(content: str) -> Any:
    text = content.strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
