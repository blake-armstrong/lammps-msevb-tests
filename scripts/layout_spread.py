#!/usr/bin/env python3
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from msevb_tests.cases import discover, find_case
from msevb_tests.compare import deviation
from msevb_tests.golden import load_golden
from msevb_tests.runner import Layout, RunError, run_case

LAYOUTS = ["1x1", "2x1", "4x1", "1x2", "2x2", "1x4", "3x2"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Report deviations from the golden for each layout.")
    parser.add_argument("cases", nargs="*", help="case names or ids (default: all)")
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    args = parser.parse_args()

    cases = [find_case(c) for c in args.cases] if args.cases else discover()
    layouts = [Layout.parse(l) for l in args.layouts.split(",")]
    print(f"{'case':40s} {'layout':>6s} {'nst':>4s} {'|dpe|':>9s} {'|dH|':>9s} {'|damp|':>9s} "
          f"{'|dF|':>9s} {'types':>5s} {'parts|dF|':>9s}")
    with tempfile.TemporaryDirectory(prefix="spread-") as tmp:
        for case in cases:
            golden = load_golden(case)
            for layout in layouts:
                try:
                    r = run_case(case, layout, Path(tmp) / f"{case.name}-{layout}")
                except RunError as exc:
                    print(f"{case.id:40s} {str(layout):>6s} RUN FAILED: {str(exc).splitlines()[0]}")
                    continue
                same_states = r.evb.keys == golden.evb.keys
                dh = deviation(r.evb.hamiltonian, golden.evb.hamiltonian, 0, 0).max_abs if same_states else np.nan
                da = deviation(r.evb.amplitudes, golden.evb.amplitudes, 0, 0).max_abs if same_states else np.nan
                df = deviation(r.atoms[0].forces, golden.atoms.forces, 0, 0).max_abs
                types_ok = np.array_equal(r.atoms[0].types, golden.atoms.types)
                parts = max(
                    (float(np.abs(a.forces - r.atoms[0].forces).max()) for a in r.atoms[1:]), default=0.0
                )
                print(
                    f"{case.id:40s} {str(layout):>6s} {r.evb.nstates:4d} {abs(r.pe[0] - golden.pe):9.2e} "
                    f"{dh:9.2e} {da:9.2e} {df:9.2e} {'ok' if types_ok else 'DIFF':>5s} {parts:9.2e}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
