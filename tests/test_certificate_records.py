"""Integrity checks for the stored CCZ computer-assisted certificate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "results" / "ccz_stopi_certificate"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_record_counts_and_aggregate_statuses_are_complete() -> None:
    """The archived symmetry-reduced cover has no missing branch records."""

    global_record = json.loads(
        (CERTIFICATE / "global_interval_certificate.json").read_text()
    )
    local_record = json.loads(
        (CERTIFICATE / "local_exact_certificate.json").read_text()
    )
    endpoint_record = json.loads(
        (CERTIFICATE / "endpoint_local_exact_certificate.json").read_text()
    )

    assert len(list((CERTIFICATE / "global_orbits").glob("orbit_*.json"))) == 20
    assert (
        len(
            list(
                (CERTIFICATE / "ideal_local_orbits").glob(
                    "ideal_orbit_*.json"
                )
            )
        )
        == 20
    )
    assert (
        len(
            list(
                (CERTIFICATE / "endpoint_local_branches").glob(
                    "endpoint_branch_*.json"
                )
            )
        )
        == 16
    )
    assert global_record["status"] == "passed_global_interval_certificate"
    assert global_record["unresolved_cells"] == 0
    assert global_record["physical_branches_covered_by_symmetry"] == 64
    assert local_record["status"] == "passed_exact_local_certificate"
    assert local_record["ideal_physical_branches"] == 64
    assert endpoint_record["status"] == (
        "passed_exact_local_endpoint_certificate"
    )
    assert endpoint_record["covered_adjacent_branches"] == 16


def test_child_record_hashes_match_the_aggregate_manifests() -> None:
    """Every stored child record has the digest claimed by its aggregate."""

    global_record = json.loads(
        (CERTIFICATE / "global_interval_certificate.json").read_text()
    )
    for name, expected in global_record["orbit_sha256"].items():
        assert _sha256(CERTIFICATE / "global_orbits" / name) == expected

    local_record = json.loads(
        (CERTIFICATE / "local_exact_certificate.json").read_text()
    )
    for name, expected in local_record["sha256"].items():
        if name.startswith("ideal_orbit_"):
            path = CERTIFICATE / "ideal_local_orbits" / name
        else:
            path = CERTIFICATE / name
        assert _sha256(path) == expected

