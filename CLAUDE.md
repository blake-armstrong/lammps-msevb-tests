# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Golden-fixture regression tests for `fix msevb` in the LAMMPS fork at `~/git/lammps-fork` (`src/MSEVB`). The tests drive a real MPI LAMMPS binary; there are no Python bindings involved.

## Commands

```sh
scripts/build_lammps.sh                       # configure + build fork -> ~/git/lammps-fork/build-msevb/lmp
cmake --build ~/git/lammps-fork/build-msevb -j8   # rebuild after editing src/MSEVB
uv sync
uv run pytest                                 # full matrix (~40 s on 10 cores)
uv run pytest -m "not slow"                   # skip one-partition-per-state layouts
uv run pytest -m bicarbonate                  # or -m examples
uv run pytest "tests/test_examples.py::test_forces_match_golden[h3o_grimme2015-2x2]"
uv run pytest -k licoo2_fermidirac_scf        # one case, all layouts
```

Maintenance scripts (run with `uv run python scripts/<name>.py`):
- `make_golden.py --all --force`: regenerate goldens at layout 2x1. It refuses if the coupling criterion fails, and records the fork commit plus `lammps_fork_src_modified`.
- `layout_spread.py [case ...]`: deviations from the golden for every layout. Use it to justify tolerance changes.
- `fd_forces.py CASE [--steps 3e-3,1e-3,3e-4] [--extra "kspace_style ewald 1e-10"]`: analytic vs finite-difference forces.
- `sync_inputs.py [--check]`: copy or verify vendored inputs against the source repos.
- `find_coupled_config.py CASE`: MD search for a frame with significant coupling; writes `data.coupled` and `[coupled_config]`.

Environment: `LMP` (binary), `MPIRUN` (default `mpirun --oversubscribe`), `MSEVB_RUN_TIMEOUT` (default 900 s; a timeout usually means partitions deadlocked), `LAMMPS_FORK`, `BICARBONATE_MSEVB`.

## Architecture

**Cases** are directories under `cases/<group>/<name>/`. The group (`examples` or `bicarbonate`) is the parent directory name. Each case holds:
- `case.toml`: source repo and files, units, mode `run0|minimize`, criterion `coupling|fermi_dirac`, thresholds, `[tolerances.<energy|hamiltonian|amplitudes|forces>]` overrides, `[search]` dynamics.
- `in.setup`: the source example with dynamics, dumps and `run` stripped.
- The vendored inputs.
- `golden.json` and `golden_forces.dat`.

Vendored inputs are copies. Change them in the source repo and run `sync_inputs.py`. Only `in.setup`, `case.toml` and `data.coupled` are maintained here.

**Run pipeline** (`msevb_tests/runner.py`):
1. Stage a temp dir with symlinks to the case files.
2. Render `in.setup`. It must contain `file msevb.json every 1` exactly once; extra fix keywords (e.g. `coupling_scale 0.0`) are appended there.
3. Write `in.test` as `variable p world 0..P-1`, `include in.setup`, the optional extra commands, then `include cases/common/<mode>.lmp`.
4. Launch `mpirun -np P*M lmp -partition PxM`.

The common tail writes `energy.<p>.txt` and `forces.<p>.dump` on every partition. Forces are `atom->f` exactly as fix msevb left it. The universe root writes `msevb.json`, and only its last record is used.

**Comparison** (`results.py`, `golden.py`, `compare.py`):
- EVB states are canonicalized by their transfer-chain keys (reference first, then sorted), because detection order can differ between domain decompositions.
- Eigenvalues are sorted, and Fermi-Dirac occupations are reordered to match.
- The JSON Hamiltonian stores the unscaled coupling, so `criteria.py` multiplies off-diagonals by `coupling_scale`.
- After a step each partition deliberately holds its own EVB state's topology. Across partitions only `pe` and forces must agree; types, charges and `f_evb` legitimately differ, and goldens use partition 0.

**Tests**: `tests/suite.py` holds the test functions and is not collected on its own. `test_examples.py` and `test_bicarbonate.py` import them and set a module-level `GROUP`. `conftest.pytest_generate_tests` parametrizes `(case, layout)` from that group:
- base layouts `1x1 2x1 4x1 1x2 2x2 1x4 3x2`;
- `slow` layouts derived from the golden's `nstates`: `Nx1` (all states in parallel) and `(N+1)x1` (idle partition).

Each `(case, layout, extra_fix_args)` is run once per session and cached, and run errors are cached too.

## Gotchas

- Any LAMMPS command that calls `lmp->init()` (e.g. `write_data`) must run on every partition. `FixMSEVB::init` makes a universe-wide collective, so `partition yes N ...` deadlocks.
- Finite differences from a single step size mislead:
  - energy noise (about 1e-6 eV in the large ionic cases) dominates at small steps;
  - unsmoothed `lj/cut` cutoffs (bicarbonate water O-O, imidazole) produce ~5e-5 eV energy jumps;
  - PPPM/Ewald at 1e-5 accuracy is not force/energy consistent.

  Sweep steps and tighten kspace via `--extra`.
- With `fermi_dirac`, the reported energy is the free energy A = Σ fλ − kT·S; the forces are −∇A.
- Tolerance changes need `layout_spread.py` evidence and a comment in `case.toml`. After intentional behavior changes in the fork, regenerate the goldens and review which values moved.
- The fork's own instructions (`~/git/lammps-fork/.claude/CLAUDE.md`) forbid AI-attribution trailers in fork commit messages.
