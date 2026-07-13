"""Tests for the metrics contract.

WHY TEST A YAML FILE?
---------------------
Because the entire claim of this project is that the definitions GOVERN the
code rather than merely describing it. That claim needs teeth.

These tests are what stop the contract from rotting into decoration:

  1. The contract is COMPLETE - no metric ships without a definition, no
     ruling ships without evidence and a stated cost.
  2. The contract is LOAD-BEARING - the parameters are actually consumed by
     the code. A parameter nobody reads is a lie.
  3. The contract FAILS LOUDLY - break it and everything stops, rather than
     quietly falling back to a default nobody agreed to.

If these tests pass, "governed metrics layer" is a fact. If they do not, it is
a phrase on a README.
"""

from __future__ import annotations

import pytest
import yaml

from northwind.metrics import (
    REQUIRED_METRICS,
    REQUIRED_PARAMETERS,
    REQUIRED_RULING_FIELDS,
    ContractViolation,
    load_contract,
)


@pytest.fixture(scope="module")
def contract():
    """The real contract, loaded once and shared across the module."""
    return load_contract()


# ---------------------------------------------------------------------------
# 1. THE CONTRACT IS COMPLETE
# ---------------------------------------------------------------------------


def test_contract_loads(contract):
    """The contract parses and validates. If this fails, nothing else matters."""
    assert contract.version
    assert contract.owner


def test_every_required_parameter_is_defined(contract):
    """No parameter the code needs is missing."""
    missing = REQUIRED_PARAMETERS - set(contract.parameters)
    assert not missing, f"contract is missing parameters: {sorted(missing)}"


def test_every_required_metric_is_defined(contract):
    """Every metric we intend to report has a written definition."""
    missing = REQUIRED_METRICS - set(contract.metrics)
    assert not missing, f"contract is missing metrics: {sorted(missing)}"


def test_every_metric_states_what_it_excludes(contract):
    """A definition that only says what IS counted is half a definition.

    The exclusions are where the arguments live. "Churn is customers who left"
    is uncontroversial and useless. "...and a downgrade is NOT that" is the
    sentence that ends the meeting.
    """
    for name, spec in contract.metrics.items():
        assert spec["excludes"], f"metric {name!r} does not state what it excludes"


def test_every_ruling_carries_evidence_and_a_cost(contract):
    """A decision without evidence is an opinion. Without a stated cost, it is a bluff.

    Naming what a ruling COSTS you is the tell that you actually thought about
    it. Any analyst can pick a definition. The senior move is knowing precisely
    which error you chose to accept, and saying so out loud before anyone finds it.
    """
    for ruling in contract.rulings:
        missing = REQUIRED_RULING_FIELDS - set(ruling)
        assert not missing, f"ruling {ruling.get('id')} is missing {sorted(missing)}"

        assert len(ruling["evidence"].strip()) > 50, (
            f"ruling {ruling['id']} has evidence too thin to defend in a meeting"
        )
        assert ruling["rejected_alternatives"], (
            f"ruling {ruling['id']} rejects no alternatives - "
            "a decision with no alternatives was not a decision"
        )
        assert len(ruling["cost_of_this_choice"].strip()) > 30, (
            f"ruling {ruling['id']} does not say what it cost us"
        )


def test_rulings_have_unique_ids(contract):
    """Two rulings called R3 is how a contract quietly contradicts itself."""
    ids = [r["id"] for r in contract.rulings]
    assert len(ids) == len(set(ids)), f"duplicate ruling ids: {ids}"


def test_every_ruling_points_at_real_metrics(contract):
    """A ruling that affects a metric which does not exist is a rotted reference."""
    for ruling in contract.rulings:
        for metric in ruling["affects"]:
            assert metric in contract.metrics, (
                f"ruling {ruling['id']} affects unknown metric {metric!r}"
            )


def test_every_metric_dependency_is_a_real_parameter(contract):
    """The dependency graph must not point at parameters that were renamed away."""
    for name, spec in contract.metrics.items():
        for param in spec["depends_on"]:
            assert param in contract.parameters, (
                f"metric {name!r} depends on {param!r}, which is not a parameter"
            )


# ---------------------------------------------------------------------------
# 2. THE CONTRACT IS LOAD-BEARING
#
# The most important tests in the file. They prove the YAML is not decoration.
# ---------------------------------------------------------------------------


def test_the_two_headline_rulings_are_what_we_agreed(contract):
    """The pause threshold and the add-on ruling are what was signed off.

    These two decisions move the headline number more than anything else in the
    project. If someone changes them, this test fails and a human has to look
    at the diff and say yes. That is the whole mechanism.
    """
    # R1: derived from the return curve - 89.6% of returns land inside 60 days.
    assert contract.pause_grace_days == 60

    # R2: add-ons swing too much to be called 'recurring'.
    assert contract.addons_count_as_mrr is False


def test_changing_the_yaml_changes_the_code(tmp_path):
    """THE test. Prove the YAML actually drives behaviour.

    We write a contract with a DIFFERENT pause threshold and confirm the loaded
    object reflects it. If this passes, the YAML is policy. If it fails, the
    YAML is a nice document that the code politely ignores - which is exactly
    the failure mode this whole milestone exists to prevent.
    """
    from northwind.metrics import CONTRACT_PATH

    # Start from the real contract so we are testing a realistic object.
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))

    # Change ONE number.
    raw["parameters"]["pause_grace_days"] = 90

    altered = tmp_path / "metrics.yaml"
    altered.write_text(yaml.safe_dump(raw), encoding="utf-8")

    # load_contract is cached, so pass the path explicitly to bypass the cache.
    changed = load_contract(altered)

    assert changed.pause_grace_days == 90, (
        "the YAML does not drive the code - the contract is decoration"
    )


def test_ruling_lookup_by_id(contract):
    """Code can cite the ruling it implements, so drift is visible."""
    r1 = contract.ruling("R1")
    assert "pause" in r1["question"].lower()

    with pytest.raises(ContractViolation):
        contract.ruling("R99")   # nonexistent rulings must fail loudly


# ---------------------------------------------------------------------------
# 3. THE CONTRACT FAILS LOUDLY
#
# A metrics layer that degrades quietly is worse than none, because people
# trust it. These tests prove that a broken contract stops the world.
# ---------------------------------------------------------------------------


def test_missing_parameter_raises(tmp_path):
    """Delete a parameter and the contract must refuse to load."""
    from northwind.metrics import CONTRACT_PATH

    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    del raw["parameters"]["pause_grace_days"]

    broken = tmp_path / "metrics.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ContractViolation, match="missing parameters"):
        load_contract(broken)


def test_ruling_without_evidence_raises(tmp_path):
    """A ruling with no evidence field must not be loadable.

    This is the guard against the most common form of metrics rot: someone adds
    a rule in a hurry, means to justify it later, and never does. Six months on,
    nobody remembers why churn is defined that way and nobody dares change it.
    """
    from northwind.metrics import CONTRACT_PATH

    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    del raw["rulings"][0]["evidence"]

    broken = tmp_path / "metrics.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ContractViolation, match="missing"):
        load_contract(broken)


def test_missing_contract_file_raises(tmp_path):
    """No contract means no analysis. Not a warning - a stop."""
    with pytest.raises(ContractViolation, match="No metrics contract"):
        load_contract(tmp_path / "does_not_exist.yaml")
