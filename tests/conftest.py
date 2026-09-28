import json

import pytest

from msevb_tests.cases import Case, discover
from msevb_tests.golden import load_golden
from msevb_tests.runner import Layout, RunError, lmp_path, run_case

LAYOUTS = ["1x1", "2x1", "4x1", "1x2", "2x2", "1x4", "3x2"]


def golden_nstates(case: Case) -> int | None:
    if not case.golden_json.exists():
        return None
    return json.loads(case.golden_json.read_text())["evb"]["nstates"]


def case_layouts(case: Case) -> list:
    params = [pytest.param(case, Layout.parse(l), id=f"{case.name}-{l}") for l in LAYOUTS]
    n = golden_nstates(case)
    if n is not None:
        for p, tag in ((n, "all-parallel"), (n + 1, "idle-partition")):
            layout = Layout(p, 1)
            if str(layout) not in LAYOUTS:
                params.append(
                    pytest.param(case, layout, id=f"{case.name}-{layout}-{tag}", marks=pytest.mark.slow)
                )
    return params


def pytest_generate_tests(metafunc):
    group = getattr(metafunc.module, "GROUP", None)
    if group is None:
        return
    cases = discover(group)
    if "layout" in metafunc.fixturenames:
        params = [p for case in cases for p in case_layouts(case)]
        metafunc.parametrize(("case", "layout"), params)
    elif "case" in metafunc.fixturenames:
        metafunc.parametrize("case", cases, ids=[c.name for c in cases])


def pytest_collection_modifyitems(config, items):
    if not lmp_path().exists():
        skip = pytest.mark.skip(reason=f"LAMMPS binary not found at {lmp_path()}; set LMP")
        for item in items:
            if "result" in getattr(item, "fixturenames", ()):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def run_cache():
    return {}


def cached_run(case, layout, run_cache, tmp_path_factory, extra_fix_args=""):
    key = (case.id, str(layout), extra_fix_args)
    if key not in run_cache:
        workdir = tmp_path_factory.mktemp(f"{case.name}-{layout}")
        try:
            run_cache[key] = run_case(case, layout, workdir, extra_fix_args)
        except RunError as exc:
            run_cache[key] = exc
    outcome = run_cache[key]
    if isinstance(outcome, RunError):
        pytest.fail(str(outcome), pytrace=False)
    return outcome


@pytest.fixture
def result(case, layout, run_cache, tmp_path_factory):
    return cached_run(case, layout, run_cache, tmp_path_factory)


@pytest.fixture
def golden(case):
    if not case.golden_json.exists():
        pytest.fail(f"{case.id} has no golden; run scripts/make_golden.py {case.name}", pytrace=False)
    return load_golden(case)
