from __future__ import annotations

import json
from pathlib import Path

from .environment import EpisodeTrace, trace_to_dict


def append_trace(path: str | Path, trace: EpisodeTrace) -> None:
    """Append one completed episode as a JSON Lines record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(trace_to_dict(trace), sort_keys=True) + "\n")
