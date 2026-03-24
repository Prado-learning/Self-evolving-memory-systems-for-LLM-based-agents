"""LLM client: local vLLM OpenAI API, SiliconFlow, and Volcengine ARK."""

from __future__ import annotations

import os
from pathlib import Path
from openai import OpenAI


def _load_env() -> dict:
    env = {}
    for candidate in [
        Path(__file__).resolve().parents[2] / ".env",
        Path.cwd() / ".env",
    ]:
        if candidate.exists():
            for line in candidate.read_text().strip().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
            break
    # Allow environment variable overrides
    for key in (
        "LOCAL_API_KEY", "LOCAL_MODEL", "LOCAL_BASE_URL",
        "ARK_API_KEY", "ARK_MODEL", "ARK_BASE_URL", "ARK_REASONING_EFFORT",
        "SF_API_KEY", "SF_MODEL", "SF_BASE_URL",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _get_client_config(env: dict) -> dict:
    """Return API config from env, preferring local -> SiliconFlow -> ARK."""
    if "LOCAL_BASE_URL" in env:
        return {
            "api_key": env.get("LOCAL_API_KEY", "EMPTY"),
            "base_url": env["LOCAL_BASE_URL"],
            "model": env.get("LOCAL_MODEL", "Qwen/Qwen3-8B"),
            "provider": "local",
        }
    if "SF_API_KEY" in env:
        return {
            "api_key": env["SF_API_KEY"],
            "base_url": env.get("SF_BASE_URL", "https://api.siliconflow.cn/v1"),
            "model": env.get("SF_MODEL", "Qwen/Qwen3-8B"),
            "provider": "siliconflow",
        }
    return {
        "api_key": env["ARK_API_KEY"],
        "base_url": env["ARK_BASE_URL"],
        "model": env["ARK_MODEL"],
        "provider": "ark",
        "reasoning_effort": env.get("ARK_REASONING_EFFORT"),
    }


class LLMClient:
    def __init__(self):
        env = _load_env()
        cfg = _get_client_config(env)
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self.model = cfg["model"]
        self.provider = cfg["provider"]
        self.reasoning_effort = cfg.get("reasoning_effort")
        self.total_tokens = 0

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            kwargs["stop"] = stop
        # ARK supports reasoning_effort; SiliconFlow/local do not
        effort = reasoning_effort or self.reasoning_effort
        if effort and self.provider == "ark":
            kwargs["reasoning_effort"] = effort
        # Qwen3 on local vLLM emits <think> by default; disable for action-only outputs.
        if self.provider == "local":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        resp = self.client.chat.completions.create(**kwargs)
        if resp.usage:
            self.total_tokens += resp.usage.total_tokens
        return resp.choices[0].message.content or ""
