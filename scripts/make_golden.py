#!/usr/bin/env python3
import argparse
import datetime
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from msevb_tests.cases import SOURCE_ROOTS, discover, find_case
from msevb_tests.criteria import evaluate_case
from msevb_tests.golden import save_golden
from msevb_tests.runner import Layout, RunError, lmp_path, mpirun_command, run_case


def command_output(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_meta(layout: Layout) -> dict:
    env, default = SOURCE_ROOTS["lammps-fork"]
    fork = Path(os.environ.get(env, default))
    help_lines = command_output([str(lmp_path()), "-h"]).splitlines()
    fft = [line.split("=", 1)[1].strip() for line in help_lines if line.startswith("FFT library")]
    mpi = command_output(mpirun_command()[:1] + ["--version"]).splitlines()
    return {
        "lammps_fork_commit": command_output(["git", "rev-parse", "HEAD"], fork).strip(),
        "lammps_fork_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], fork).strip(),
        "lammps_fork_src_modified": bool(
            command_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "src"], fork).strip()
        ),
        "lammps_version": next((l.strip() for l in help_lines if l.startswith("Large-scale")), ""),
        "fft_library": fft[0] if fft else "",
        "mpirun": mpi[0] if mpi else "",
        "layout": str(layout),
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate golden energies and forces.")
    parser.add_argument("cases", nargs="*", help="case names or ids")
    parser.add_argument("--all", action="store_true", help="every case")
    parser.add_argument("--layout", default="2x1")
    parser.add_argument("--force", action="store_true", help="overwrite existing goldens")
    args = parser.parse_args()
    if not args.cases and not args.all:
        parser.error("name cases or pass --all")

    cases = discover() if args.all else [find_case(c) for c in args.cases]
    layout = Layout.parse(args.layout)
    meta = build_meta(layout)
    failed = 0
    with tempfile.TemporaryDirectory(prefix="golden-") as tmp:
        for case in cases:
            if case.golden_json.exists() and not args.force:
                print(f"{case.id}: golden exists, skipping (use --force)")
                continue
            try:
                result = run_case(case, layout, Path(tmp) / case.name)
            except RunError as exc:
                print(f"{case.id}: run failed\n{exc}")
                failed += 1
                continue
            report = evaluate_case(case, result.evb)
            if not report.ok:
                print(f"{case.id}: refusing to write golden: {report.summary()}")
                failed += 1
                continue
            save_golden(case, result, {**meta, "criterion": report.metrics})
            print(f"{case.id}: pe={result.pe[0]:.12g} {report.summary()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
