"""Lazy public API for model clients.

Avoid importing every optional backend dependency at package import time.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "BaseModelClient",
    "GLMClient",
    "GeminiClient",
    "GemmaClient",
    "KimiThinkingClient",
    "Molmo2Client",
    "OpenAIClient",
    "QwenClient",
]

_EXPORTS = {
    "BaseModelClient": ("topreward.clients.base", "BaseModelClient"),
    "GLMClient": ("topreward.clients.glm", "GLMClient"),
    "GeminiClient": ("topreward.clients.gemini", "GeminiClient"),
    "GemmaClient": ("topreward.clients.gemma", "GemmaClient"),
    "KimiThinkingClient": ("topreward.clients.kimi", "KimiThinkingClient"),
    "Molmo2Client": ("topreward.clients.molmo", "Molmo2Client"),
    "OpenAIClient": ("topreward.clients.openai", "OpenAIClient"),
    "QwenClient": ("topreward.clients.qwen", "QwenClient"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
