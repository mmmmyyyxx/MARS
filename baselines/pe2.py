from __future__ import annotations

from mars_core.mars_runner import run_candidate_method


def run(**kwargs):
    return run_candidate_method(method="pe2", **kwargs)
