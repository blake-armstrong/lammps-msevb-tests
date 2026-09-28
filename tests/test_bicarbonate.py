import pytest

from suite import (
    test_energy_matches_golden,
    test_evb_matches_golden,
    test_forces_match_golden,
    test_golden_satisfies_criterion,
    test_partitions_identical,
    test_run_satisfies_criterion,
)

GROUP = "bicarbonate"
pytestmark = pytest.mark.bicarbonate
