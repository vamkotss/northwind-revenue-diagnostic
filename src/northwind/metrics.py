"""The metrics contract: load it, validate it, and refuse to run without it.

WHY THIS MODULE EXISTS
----------------------
A metrics document that the code does not read is decoration. Within six weeks
the dashboard says one thing and the doc says another, and nobody notices until
a board meeting.

This module makes docs/metrics/metrics.yaml LOAD-BEARING. Every downstream
calculation asks this module for its parameters. Change pause_grace_days from
60 to 90 in the YAML and the churn number actually changes - because there is
nowhere else for the code to get that number from.

That is the difference between documenting a definition and governing one.

THE CONTRACT
------------
  - Every parameter used by the code must exist in the YAML.
  - Every metric must declare which parameters it depends on.
  - Every ruling must carry evidence and name what it cost us.

If any of that is missing, this module raises. It does not warn and carry on;
a metrics layer that degrades quietly is worse than none, because people trust
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Where the contract lives. Resolved relative to this file, so it works no
# matter which directory you happen to run from.
#   parents[0] = northwind/   parents[1] = src/   parents[2] = repo root
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "docs" / "metrics" / "metrics.yaml"

# Parameters the code genuinely relies on. If the YAML drops one, we fail loudly
# rather than silently substituting a default - a default is an undocumented
# decision, and undocumented decisions are the whole problem we are solving.
REQUIRED_PARAMETERS = {
    "pause_grace_days",
    "addons_count_as_mrr",
    "winback_window_days",
    "refund_attribution",
    "restate_history_on_correction",
    "currency",
}

# Metrics that must be defined before any analysis is allowed to run.
REQUIRED_METRICS = {
    "mrr",
    "arr",
    "addon_revenue",
    "logo_churn",
    "revenue_churn",
    "contraction",
    "expansion",
    "gross_revenue_retention",
    "net_revenue_retention",
}

# Every ruling must answer all of these. "We decided X" without evidence is an
# opinion; with evidence it is a ruling. The difference matters in a room with
# a CFO in it.
REQUIRED_RULING_FIELDS = {
    "id",
    "question",
    "ruling",
    "evidence",
    "rejected_alternatives",
    "cost_of_this_choice",
    "affects",
}


class ContractViolation(Exception):
    """Raised when the metrics contract is incomplete or internally inconsistent.

    Deliberately fatal. If the definitions are broken, every number computed
    downstream is untrustworthy, and producing them anyway would be worse than
    producing nothing.
    """


@dataclass(frozen=True)
class MetricsContract:
    """The parsed, validated metrics contract.

    frozen=True makes this immutable: once loaded, no code can quietly reach in
    and change pause_grace_days at runtime. If you want a different value, you
    change the YAML - which is a diff, which is reviewable, which is the point.
    """

    version: str
    effective_date: str
    owner: str
    status: str
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    rulings: list[dict[str, Any]]
    out_of_scope: list[str]

    # --- Convenience accessors for the parameters used most often -----------
    # These exist so that calling code reads as English rather than as
    # dictionary lookups, and so that a typo in a key name fails at import
    # rather than silently returning None halfway through a calculation.

    @property
    def pause_grace_days(self) -> int:
        """Days a subscription may stay paused before it counts as churn."""
        return int(self.parameters["pause_grace_days"])

    @property
    def addons_count_as_mrr(self) -> bool:
        """Whether usage add-on charges are folded into MRR."""
        return bool(self.parameters["addons_count_as_mrr"])

    @property
    def winback_window_days(self) -> int:
        """Days within which a returning customer is the same logo, not a new one."""
        return int(self.parameters["winback_window_days"])

    @property
    def refund_attribution(self) -> str:
        """Which month a refund reduces revenue in."""
        return str(self.parameters["refund_attribution"])

    @property
    def restate_history_on_correction(self) -> bool:
        """Whether a backdated correction rewrites the historical number."""
        return bool(self.parameters["restate_history_on_correction"])

    def ruling(self, ruling_id: str) -> dict[str, Any]:
        """Fetch a single ruling by id, e.g. 'R1'.

        Used when a piece of code wants to cite the ruling it is implementing -
        which makes the code self-documenting and, more usefully, makes it
        obvious when a ruling changes but the code does not.
        """
        for r in self.rulings:
            if r["id"] == ruling_id:
                return r
        raise ContractViolation(f"No ruling with id {ruling_id!r} in the contract")


def _validate(raw: dict[str, Any]) -> None:
    """Check the contract is complete and internally consistent. Raise if not."""
    # --- Top-level structure ------------------------------------------------
    for key in ["version", "parameters", "metrics", "rulings"]:
        if key not in raw:
            raise ContractViolation(f"Contract is missing the top-level key {key!r}")

    # --- Every required parameter is present --------------------------------
    missing_params = REQUIRED_PARAMETERS - set(raw["parameters"])
    if missing_params:
        raise ContractViolation(f"Contract is missing parameters: {sorted(missing_params)}")

    # --- Every required metric is defined ------------------------------------
    missing_metrics = REQUIRED_METRICS - set(raw["metrics"])
    if missing_metrics:
        raise ContractViolation(f"Contract is missing metrics: {sorted(missing_metrics)}")

    # --- Every metric is fully specified -------------------------------------
    for name, spec in raw["metrics"].items():
        for field in ["definition", "formula", "grain", "excludes", "depends_on"]:
            if field not in spec:
                raise ContractViolation(f"Metric {name!r} is missing {field!r}")

        # A metric cannot depend on a parameter that does not exist. This catches
        # the classic failure: someone renames a parameter and the dependency
        # graph silently rots.
        for param in spec["depends_on"]:
            if param not in raw["parameters"]:
                raise ContractViolation(
                    f"Metric {name!r} depends on {param!r}, which is not a parameter"
                )

    # --- Every ruling is fully argued ----------------------------------------
    for ruling in raw["rulings"]:
        missing_fields = REQUIRED_RULING_FIELDS - set(ruling)
        if missing_fields:
            rid = ruling.get("id", "<no id>")
            raise ContractViolation(f"Ruling {rid} is missing: {sorted(missing_fields)}")

        # A ruling must name the metrics it affects, and those must be real.
        for metric in ruling["affects"]:
            if metric not in raw["metrics"]:
                raise ContractViolation(
                    f"Ruling {ruling['id']} affects {metric!r}, which is not a defined metric"
                )

    # --- Ruling ids are unique -----------------------------------------------
    ids = [r["id"] for r in raw["rulings"]]
    if len(ids) != len(set(ids)):
        raise ContractViolation(f"Duplicate ruling ids in contract: {ids}")


@lru_cache(maxsize=1)
def load_contract(path: Path | None = None) -> MetricsContract:
    """Load, validate, and return the metrics contract.

    Cached, because the contract is read on every metric calculation and parsing
    YAML a thousand times would be silly. lru_cache means it is parsed once per
    process, then handed out.
    """
    contract_path = path or CONTRACT_PATH

    if not contract_path.exists():
        raise ContractViolation(
            f"No metrics contract found at {contract_path}. "
            "Analysis does not run without agreed definitions."
        )

    # safe_load refuses to execute arbitrary Python embedded in the YAML.
    # Always use safe_load. Never use yaml.load on a file you did not write.
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

    _validate(raw)

    return MetricsContract(
        version=raw["version"],
        effective_date=raw.get("effective_date", "unknown"),
        owner=raw.get("owner", "unknown"),
        status=raw.get("status", "unknown"),
        parameters=raw["parameters"],
        metrics=raw["metrics"],
        rulings=raw["rulings"],
        out_of_scope=raw.get("out_of_scope", []),
    )


def describe() -> str:
    """Print a human summary of the contract. Useful in a demo, and in a README."""
    c = load_contract()

    lines = [
        f"Northwind Metrics Contract v{c.version}  ({c.status})",
        f"Owner: {c.owner}   Effective: {c.effective_date}",
        "",
        "PARAMETERS - every number below is a judgement call, each defended in a ruling:",
    ]
    for key, value in c.parameters.items():
        lines.append(f"  {key:32s} {value}")

    lines += ["", f"METRICS DEFINED: {len(c.metrics)}", "", "RULINGS:"]
    for r in c.rulings:
        lines.append(f"  [{r['id']}] {r['question']}")
        lines.append(f"       -> {r['ruling'].strip()}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
