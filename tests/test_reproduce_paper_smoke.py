import subprocess
import sys


def test_reproduce_paper_smoke_dryrun():
    result = subprocess.run(
        [
            sys.executable,
            "reproduce_paper.py",
            "--preset",
            "smoke",
            "--dry-run",
            "--max-samples",
            "2",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Summary:" in result.stdout
    validation = subprocess.run(
        [sys.executable, "validate_reproduction_outputs.py", "--latest"],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert validation.returncode == 0, validation.stderr + validation.stdout
