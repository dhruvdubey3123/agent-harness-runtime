"""Turn yesterday's run logs into today's procedures.

`distill` reads finished runs, groups them by task kind, and when the same tool sequence
has succeeded at least `min_support` times, writes it out as a skill file. `load_skills_index`
puts the matching skill into the system prompt of the next run of that kind.

It is intentionally simple and inspectable: plain markdown files and a JSON index you can
read, edit or delete. Nothing is learned that a person cannot see.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .store import RunStore


def distill(store: RunStore, out_dir: str | Path, min_support: int = 2) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_kind: dict[str, Counter] = defaultdict(Counter)
    example_args: dict[tuple, list[dict]] = {}
    for run in store.runs(status="done"):
        steps = [s for s in store.steps(run["id"]) if s["status"] == "done"]
        if not steps:
            continue
        seq = tuple(s["tool"] for s in steps)
        kind = run["task_kind"] or "general"
        by_kind[kind][seq] += 1
        example_args.setdefault((kind, seq), [sorted(s["args"].keys()) for s in steps])

    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for kind, counts in by_kind.items():
        seq, support = counts.most_common(1)[0]
        if support < min_support:
            continue
        arg_keys = example_args[(kind, seq)]
        lines = [f"---", f"name: {kind}", f"support: {support}", f"---", "",
                 f"# {kind}", "", f"Succeeded {support} times with this tool sequence:", ""]
        lines += [f"{i}. `{t}` with args {', '.join(k) or '(none)'}" for i, (t, k) in enumerate(zip(seq, arg_keys), 1)]
        (out / f"{kind}.md").write_text("\n".join(lines) + "\n")
        index[kind] = {"file": f"{kind}.md", "support": support, "steps": list(seq)}
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index


def load_skills_index(skills_dir: str | Path | None, task_kind: str | None) -> str:
    if not skills_dir or not task_kind:
        return ""
    p = Path(skills_dir) / "index.json"
    if not p.exists():
        return ""
    entry = json.loads(p.read_text()).get(task_kind)
    if not entry:
        return ""
    return f"{task_kind}: " + " -> ".join(entry["steps"]) + f" (worked {entry['support']} times)"
