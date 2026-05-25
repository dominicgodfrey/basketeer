"""Prompt template loader.

All LLM prompts live in this directory as `.md` files. Business logic loads them
through `load_prompt(name)` rather than embedding multi-line strings inline.
Keeping prompts as separate files lets us version, diff, and reformat them
without touching Python code, and makes A/B testing prompt variants tractable.
"""

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Return the contents of `<name>.md` from the prompts directory.

    Raises FileNotFoundError if the prompt doesn't exist — surfacing missing
    prompts at startup is preferable to silent fallback.
    """
    path = _PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


__all__ = ["load_prompt"]
