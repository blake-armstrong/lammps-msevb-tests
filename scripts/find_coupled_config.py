#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np

from msevb_tests.cases import find_case, load_case
from msevb_tests.criteria import evaluate_case
from msevb_tests.results import canonical_record
from msevb_tests.runner import Layout, RunError, launch, run_case, stage

COUPLED_DATA = "data.coupled"
READ_DATA = re.compile(r"^(read_data\s+)(\S+)", re.MULTILINE)


def search_input(layout: Layout, dynamics: str, every: int, chunks: int) -> str:
    worlds = " ".join(str(p) for p in range(layout.partitions))
    return f"""variable p world {worlds}
include in.setup
{dynamics}
thermo_style custom step pe f_evb
thermo {every}
variable i loop {chunks}
label chunk
run {every} post no
write_data frame.$i.${{p}} nocoeff
next i
jump SELF chunk
"""


def frame_records(workdir: Path, every: int, chunks: int) -> dict[int, dict]:
    records = json.loads((workdir / "msevb.json").read_text())["timesteps"]
    by_step = {}
    for rec in records:
        by_step[rec["timestep"]] = rec
    return {i: by_step[i * every] for i in range(1, chunks + 1) if i * every in by_step}


def second_amplitude(evb) -> float:
    amps = np.sort(evb.amplitudes)[::-1]
    return float(amps[1]) if amps.size > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find a configuration with non-zero EVB coupling by running the source dynamics."
    )
    parser.add_argument("case")
    parser.add_argument("--layout", default="2x1")
    parser.add_argument("--every", type=int, default=10, help="steps between saved frames")
    parser.add_argument("--chunks", type=int, default=100, help="number of saved frames")
    parser.add_argument("--verify", type=int, default=5, help="candidates to re-check with run 0")
    parser.add_argument("--keep", action="store_true", help="keep the search directory")
    args = parser.parse_args()

    case = find_case(args.case)
    layout = Layout.parse(args.layout)
    toml = tomllib.loads(case.path.joinpath("case.toml").read_text())
    if "coupled_config" in toml:
        print(f"{case.id}: already uses {toml['coupled_config']['data']}; remove [coupled_config] to redo")
        return 0
    spec = toml.get("search")
    if not spec:
        print(f"{case.id}: case.toml has no [search] table with the source dynamics")
        return 2

    workdir = Path(tempfile.mkdtemp(prefix=f"search-{case.name}-"))
    stage(case, workdir)
    (workdir / "in.search").write_text(search_input(layout, spec["dynamics"], args.every, args.chunks))
    print(f"{case.id}: running {args.chunks} x {args.every} steps in {workdir}")
    launch(workdir, layout, "in.search")

    ranked = []
    for i, rec in frame_records(workdir, args.every, args.chunks).items():
        evb = canonical_record(rec)
        report = evaluate_case(case, evb)
        if report.ok:
            ranked.append((second_amplitude(evb), i, report))
    ranked.sort(key=lambda t: t[0], reverse=True)
    print(f"{len(ranked)} frame(s) pass the criterion in the trajectory")

    original = case.setup_file.read_text()
    source_data = READ_DATA.search(original).group(2)
    for amp2, i, report in ranked[: args.verify]:
        frame = workdir / f"frame.{i}.0"
        trial_dir = workdir / f"verify.{i}"
        trial_dir.mkdir()
        shutil.copy2(frame, trial_dir / COUPLED_DATA)
        for src in case.input_files() + [case.path / "case.toml"]:
            if src.name != COUPLED_DATA:
                (trial_dir / src.name).symlink_to(src)
        (trial_dir / "in.setup").write_text(READ_DATA.sub(rf"\g<1>{COUPLED_DATA}", original, count=1))
        trial = load_case(trial_dir)
        try:
            result = run_case(trial, layout, workdir / f"run.{i}")
        except RunError as exc:
            print(f"frame {i}: run 0 failed: {exc}")
            continue
        check = evaluate_case(case, result.evb)
        print(f"frame {i} (step {i * args.every}, 2nd amplitude {amp2:.4g}): run 0 {check.summary()}")
        if check.ok:
            shutil.copy2(frame, case.path / COUPLED_DATA)
            case.setup_file.write_text(trial_dir.joinpath("in.setup").read_text())
            with case.path.joinpath("case.toml").open("a") as fh:
                fh.write(
                    f"\n[coupled_config]\n"
                    f'data = "{COUPLED_DATA}"\n'
                    f'from_data = "{source_data}"\n'
                    f"step = {i * args.every}\n"
                    f'layout = "{layout}"\n'
                )
            print(f"{case.id}: vendored frame {i} as {COUPLED_DATA}")
            if not args.keep:
                shutil.rmtree(workdir)
            return 0

    print(f"{case.id}: no verified coupled frame; try more --chunks or a different --every")
    return 1


if __name__ == "__main__":
    sys.exit(main())
