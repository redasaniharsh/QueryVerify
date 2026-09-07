"""Lightweight trace helper for the request pipeline.

Every pipeline stage appends a small dict to a shared trace list so the
frontend can render a step-by-step timeline of what happened.
"""


def add(trace: list, step: str, detail: str, duration_s: float) -> None:
    """Append a single trace entry.  No-ops when *trace* is None."""
    if trace is not None:
        trace.append(
            {"step": step, "detail": detail, "duration_s": round(duration_s, 2)}
        )
