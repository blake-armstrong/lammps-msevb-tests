# lammps-msevb-tests

Golden-fixture regression tests for `fix msevb` in the LAMMPS fork
(`~/git/lammps-fork`, package `src/MSEVB`).

Each case is a single-point energy and force evaluation (`run 0`, or 10 fixed CG
iterations for the minimisation example). Every run is checked for three things:

- **It matches the golden:** `pe`, `f_evb`, the EVB Hamiltonian, eigenvalues,
  amplitudes, Fermi-Dirac occupations, atom types and charges, and per-atom forces.
- **It doesn't depend on layout:** it is repeated for partition counts and
  ranks-per-partition `1x1 2x1 4x1 1x2 2x2 1x4 3x2`, plus (marked `slow`) one
  partition per EVB state and one partition more than that.
- **It is really coupled:** the configuration has non-zero EVB coupling. That means
  at least 2 states, effective off-diagonal |H_ij| ≥ `min_coupling`, a
  coupling-induced ground-state lowering of at least `min_lowering`, and at least
  2 states with amplitude ≥ `min_weight`. The LiCoO2 examples use `coupling none`,
  so they require non-trivial Fermi-Dirac mixing instead.

## Setup

```sh
scripts/build_lammps.sh            # builds ~/git/lammps-fork/build-msevb/lmp (MPI, KISS FFT)
uv sync
```

Environment variables:

| variable | default | meaning |
|---|---|---|
| `LMP` | `~/git/lammps-fork/build-msevb/lmp` | LAMMPS binary |
| `MPIRUN` | `mpirun --oversubscribe` | MPI launcher |
| `MSEVB_RUN_TIMEOUT` | `900` | seconds before a run is treated as hung |
| `LAMMPS_FORK` / `BICARBONATE_MSEVB` | `~/git/...` | where `sync_inputs.py` copies inputs from |

## Running

```sh
uv run pytest                      # everything (~40 s on 10 cores)
uv run pytest -m "not slow"        # skip one-partition-per-state layouts
uv run pytest -m bicarbonate       # or -m examples
```

## Layout

```
cases/common/{run0,minimize}.lmp   shared tail: run, then write energy.<p>.txt and forces.<p>.dump
cases/examples/<name>/             vendored from examples/PACKAGES/msevb
cases/bicarbonate/bicarbonate_water/  vendored from bicarbonate-msevb/msevb (coord.lmp)
  case.toml                        source, units, mode, criterion, thresholds, tolerance overrides
  in.setup                         the example input without dynamics/dumps/run
  golden.json, golden_forces.dat   the fixtures
msevb_tests/                       harness (runner, parsing, criteria, comparison, finite differences)
tests/                             pytest suite (suite.py is shared by test_examples/test_bicarbonate)
scripts/                           maintenance tools, below
```

Forces are read from `atom->f` after `fix msevb` has mixed them (`write_dump` after
the run, which doesn't recompute forces). EVB quantities come from the step record
of `file msevb.json every 1`. States are matched by their transfer chains rather
than by index, so a different detection order can't produce a false mismatch.

## Scripts

- `sync_inputs.py [--check]`: copy each case's `files` from its source repo, or
  report drift.
- `find_coupled_config.py CASE`: when the source data file has no significant
  coupling, run the source dynamics (from `[search]` in `case.toml`), pick the
  frame with the strongest mixing, re-verify it at `run 0`, and vendor it as
  `data.coupled`. It records provenance in `[coupled_config]`. This was used for
  `bazro3h_raiteri2011` and `bicarbonate_water`.
- `make_golden.py --all [--force]`: regenerate goldens (layout 2x1). It refuses to
  write a golden that fails its criterion, and records the fork commit, LAMMPS
  version, FFT library and MPI version.
- `layout_spread.py`: deviation from the golden for every case × layout. Use it to
  justify tolerance changes.
- `fd_forces.py CASE`: compare analytic forces in several layouts against central
  finite differences of the potential energy. Useful options:
  - `--steps` sweeps step sizes and reports the best agreement.
  - `--extra "kspace_style ewald 1e-10"` removes long-range solver inconsistency.

## Tolerances

Defaults are in `msevb_tests/compare.py`: energy and H `rtol 1e-10, atol 1e-7`;
amplitudes `atol 1e-7`; forces `rtol 1e-7, atol 1e-7`. Observed spread across
layouts is ≤ 1.5e-9 in energy and ≤ 1.5e-11 in force, except for LiCoO2, whose
documented per-case overrides are in `case.toml`.

## Known issues found by this suite

- **Fixed in the fork, not yet committed:** in-fix force recomputes (serial
  states, SCF topology loop) cleared forces on owned atoms only. With `newton on`,
  stale ghost forces leaked into the serial-state forces. Energies were right,
  but forces depended on the partition count.
- **Fixed in the fork, not yet committed:** minimization with more than one rank
  per partition depended on the layout. `FixMSEVB::min_pre_force` ran
  `pbc`/`exchange`/`borders` on evaluations where the minimizer had not
  reneighbored, so atoms migrated between ranks while the minimizer kept using
  stale per-atom vectors (`Min::energy_force` only calls `reset_vectors()` when it
  reneighbors itself). `FixMSEVB::min_setup` now turns off the distance check, so
  the minimizer reneighbors on every force evaluation; `Min::cleanup` restores
  the user's settings. Full example minimizations (about 440 CG iterations) now
  differ across layouts by at most 6e-4 eV. That is smaller than the 1.9e-3 eV
  produced by nudging one atom by 1e-12 Å on a single layout, i.e. ordinary
  minimizer sensitivity to roundoff.
- **Fixed in the fork, not yet committed:** with `fermi_dirac`, the forces are
  −∇A of the free energy A = Σ fλ − kT·S, but the fix used to report Σ fλ as
  the energy. It now reports A. Forces, occupations and trajectories are
  unchanged; for the LiCoO2 goldens the PE shifted by −0.049 eV. The JSON
  `fermi_dirac` block now has `free_energy`, `energy` and `entropy` in place of
  `E_mixed`.
