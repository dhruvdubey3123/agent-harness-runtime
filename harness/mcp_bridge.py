"""Expose an agent's tools as an MCP server that contains *only* that agent's allowed tools.

One server per agent policy is the least-privilege version of MCP: a client connected to
the "reader" server cannot even see that write tools exist.

    python -m harness.mcp_bridge --workspace ./ws --allow read_file,list_files
"""
from __future__ import annotations

import argparse
import inspect

from mcp.server.mcpserver import MCPServer

from .sandbox import Policy, Sandbox
from .tools import ToolRegistry, workspace_tools
from .types import ToolCall


def build_server(registry: ToolRegistry, policy: Policy, name: str = "harness-tools") -> MCPServer:
    sandbox = Sandbox(registry, policy)
    server = MCPServer(name, instructions="Tools are executed through the harness sandbox.")
    for tool_name in sorted(policy.allowed_tools):
        tool = registry.get(tool_name)
        if tool is None:
            continue

        def make(tn: str, args_model):
            def call(**kwargs) -> str:
                return str(sandbox.execute(ToolCall(tn, kwargs)))
            # Give the wrapper the tool's real signature so MCP publishes a proper input schema.
            params = [inspect.Parameter(f, inspect.Parameter.KEYWORD_ONLY, annotation=info.annotation)
                      for f, info in args_model.model_fields.items()]
            call.__signature__ = inspect.Signature(params, return_annotation=str)
            call.__annotations__ = {f: info.annotation for f, info in args_model.model_fields.items()} | {"return": str}
            return call

        server.add_tool(make(tool_name, tool.args_model), name=tool_name, description=tool.description)
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--allow", default="read_file,list_files")
    a = ap.parse_args()
    build_server(workspace_tools(a.workspace), Policy(set(a.allow.split(",")))).run()


if __name__ == "__main__":
    main()
