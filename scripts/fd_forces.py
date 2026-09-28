#!/usr/bin/env python3
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from msevb_tests.cases import find_case
from msevb_tests.finite_diff import fd_force, reactive_atoms
from msevb_tests.runner import Layout, run_case


def resolve_layouts(tokens: list[str], nstates: int) -> list[Layout]:
    layouts = []
    for token in tokens:
        layout = Layout(nstates, 1) if token == "all" else Layout.parse(token)
        if layout not in layouts:
            layouts.append(layout)
    return layouts


def fmt(vector) -> str:
    return np.array2string(np.asarray(vector), precision=6, floatmode="fixed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check analytic fix msevb forces against central finite differences of the energy. "
        "Each atom is differentiated at every --steps value and the best agreement is reported, since "
        "a single step can be spoiled by energy noise (small steps) or by potential cutoffs that are "
        "not smoothed (large steps)."
    )
    parser.add_argument("case")
    parser.add_argument("--atoms", default="", help="comma-separated atom ids to check")
    parser.add_argument("--reactive", type=int, default=3, help="number of reactive (X/H/Y) atoms to add")
    parser.add_argument("--divergent", type=int, default=3,
                        help="number of atoms where the compared layouts disagree most to add")
    parser.add_argument("--compare", default="1x1,2x1,all",
                        help="layouts whose analytic forces are checked; 'all' = one partition per state")
    parser.add_argument("--fd-layout", default="1x1", help="layout used for the displaced energy runs")
    parser.add_argument("--steps", default="3e-3,1e-3,3e-4",
                        help="comma-separated displacements (distance units)")
    parser.add_argument("--tol", type=float, default=1.0e-3, help="max |F - F_fd| accepted (force units)")
    parser.add_argument("--extra", default="",
                        help="LAMMPS commands run after in.setup for every run, e.g. a tighter kspace_style")
    parser.add_argument("--keep", action="store_true", help="keep the run directories")
    args = parser.parse_args()

    case = find_case(args.case)
    steps = [float(s) for s in args.steps.split(",")]
    workdir = Path(tempfile.mkdtemp(prefix=f"fd-{case.name}-"))
    probe = run_case(case, Layout(1, 1), workdir / "probe", extra_commands=args.extra, mode="run0")
    compare = resolve_layouts(args.compare.split(","), probe.evb.nstates)
    fd_layout = resolve_layouts([args.fd_layout], probe.evb.nstates)[0]

    analytic = {
        str(l): run_case(case, l, workdir / f"ref-{l}", extra_commands=args.extra, mode="run0")
        for l in compare
    }
    first = analytic[str(compare[0])]
    reference = analytic.get(str(fd_layout), first)
    ids = first.atoms[0].ids
    forces = {k: r.atoms[0].forces for k, r in analytic.items()}

    atoms = [int(a) for a in args.atoms.split(",") if a]
    atoms += [a for a in reactive_atoms(first)[: args.reactive] if a not in atoms]
    if len(forces) > 1 and args.divergent:
        spread = np.max([np.abs(f - forces[str(compare[0])]).max(axis=1) for f in forces.values()], axis=0)
        atoms += [int(a) for a in ids[np.argsort(spread)[::-1][: args.divergent]] if int(a) not in atoms]

    print(f"{case.id}: {first.evb.nstates} states; d(pe)/dx at layout {fd_layout}; "
          f"steps {', '.join(f'{s:g}' for s in steps)}; analytic layouts {', '.join(forces)}")
    worst = {k: 0.0 for k in forces}
    for atom_id in atoms:
        i = int(np.flatnonzero(ids == atom_id)[0])
        fds = [
            fd_force(case, fd_layout, workdir / "fd", atom_id, reference, s, args.extra)
            for s in steps
        ]
        print(f"atom {atom_id}")
        for fd in fds:
            note = "" if fd.states_stable else "  (state set or types changed under displacement)"
            print(f"  FD h={fd.step:<8g} {fmt(fd.force)}{note}")
        for k, f in forces.items():
            errs = [float(np.abs(f[i] - fd.force).max()) for fd in fds]
            best = min(errs)
            worst[k] = max(worst[k], best)
            print(f"  {k:>12s} {fmt(f[i])}  best |F - F_fd| = {best:.2e} "
                  f"(per step: {', '.join(f'{e:.1e}' for e in errs)})")

    print("max over atoms of best |F - F_fd|:")
    failed = False
    for k, err in worst.items():
        bad = err > args.tol
        failed |= bad
        print(f"  {k:>7s} {err:.2e}{'  EXCEEDS tol' if bad else ''}")
    if args.keep:
        print(f"run directories kept in {workdir}")
    else:
        shutil.rmtree(workdir)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
