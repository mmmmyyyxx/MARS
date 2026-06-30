from __future__ import annotations

from mars_core.mars_runner import run_direct_method


def run(**kwargs):
    return run_direct_method(method="cot_zs", **kwargs)
