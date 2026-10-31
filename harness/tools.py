"""Tool registry. A tool declares its argument schema and whether calling it twice is safe.

`idempotent` matters for crash recovery: if the process dies after a tool started but before
its result was recorded, the harness may re-run an idempotent tool, but must stop and ask a
human before re-running one that is not (it might already have sent the email, written the
row, or charged the card).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]
    idempotent: bool = True
    side_effects: bool = False

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.args_model.model_json_schema()}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def tool(self, name: str, description: str, args_model: type[BaseModel], idempotent: bool = True,
             side_effects: bool = False) -> Callable:
        def deco(fn: Callable) -> Callable:
            self.register(Tool(name, description, args_model, fn, idempotent, side_effects))
            return fn
        return deco


# ---------------------------------------------------------------- built-in workspace tools

class NoArgs(BaseModel):
    pass


class PathArgs(BaseModel):
    path: str


class WriteArgs(BaseModel):
    path: str
    content: str


class AppendArgs(BaseModel):
    path: str
    line: str


def workspace_tools(root: str | Path) -> ToolRegistry:
    """File tools jailed to `root`. The sandbox also enforces the jail; this is defence in depth."""
    base = Path(root).resolve()
    reg = ToolRegistry()

    def _p(rel: str) -> Path:
        p = (base / rel).resolve()
        if base != p and base not in p.parents:
            raise PermissionError(f"path escapes workspace: {rel}")
        return p

    @reg.tool("read_file", "Read a UTF-8 text file from the workspace.", PathArgs)
    def read_file(path: str) -> str:
        return _p(path).read_text()

    @reg.tool("list_files", "List files in the workspace (recursive).", NoArgs)
    def list_files() -> list[str]:
        return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())

    @reg.tool("write_file", "Create or overwrite a file. Safe to repeat with the same content.", WriteArgs,
              idempotent=True, side_effects=True)
    def write_file(path: str, content: str) -> str:
        p = _p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    @reg.tool("append_line", "Append one line to a file. NOT safe to repeat: each call adds a line.", AppendArgs,
              idempotent=False, side_effects=True)
    def append_line(path: str, line: str) -> str:
        p = _p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(line + "\n")
        return f"appended to {path}"

    return reg
