"""DeepSeek V3 client via Volcengine ARK API."""

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
    for key in ("ARK_API_KEY", "ARK_MODEL", "ARK_BASE_URL"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


class LLMClient:
    def __init__(self):
        env = _load_env()
        self.client = OpenAI(
            api_key=env["ARK_API_KEY"],
            base_url=env["ARK_BASE_URL"],
        )
        self.model = env["ARK_MODEL"]
        self.total_tokens = 0

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        if resp.usage:
            self.total_tokens += resp.usage.total_tokens
        return resp.choices[0].message.content or ""
