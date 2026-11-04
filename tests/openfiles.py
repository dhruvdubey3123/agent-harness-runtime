"""Which SQLite files this process still holds open.

A store that outlives the directory it lives in is invisible on Linux, where an open file can be
deleted, and fatal on Windows, where it cannot: the benchmarks clean up with TemporaryDirectory,
so a leaked connection turns into PermissionError: [WinError 32] for everyone who runs the tests
there. CI is Linux, so the guard has to be explicit rather than left to the operating system.
"""
from __future__ import annotations

import os
from pathlib import Path

FD_DIR = Path("/proc/self/fd")
available = FD_DIR.exists()


def open_db_files() -> set[str]:
    out = set()
    if not available:
        return out
    for fd in FD_DIR.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:          # the descriptor went away while we were looking
            continue
        # A descriptor whose file is already unlinked reads as "/path/to/x.db (deleted)", and that
        # is exactly the case worth catching: on Linux the delete succeeded anyway.
        name = target.removesuffix(" (deleted)")
        if name.endswith(".db"):
            out.add(target)
    return out
