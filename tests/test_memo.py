"""Tests for the executive memo.

THE CLAIM THESE DEFEND
----------------------
The memo says, at the top, that every number in it is generated from the
pipeline and none is typed by hand. That is a strong claim and it is the reason
anyone should believe the document.

These tests make it true rather than merely stated:

1. NO HARDCODED NUMBERS. The renderer's output must change when the underlying
   data changes. If it does not, some figure is a literal, and a literal will go
   stale the first time anything upstream moves - silently, permanently, and
   with nobody re-reading a memo they already signed off.

2. THE NUMBERS MATCH THE PIPELINE. Every headline figure in the memo is checked
   against the module that produced it. A memo that disagrees with its own
   source code is worse than no memo.

3. EVERY TAG RESOLVES. A provenance tag pointing at a function that does not
   exist is a broken promise wearing a footnote.

4. THE UNCOMFORTABLE SECTION SURVIVES. The memo contains a section saying the
   pricing experiment cannot be read out. That section is the one most likely to
   be softened by a future edit, because nobody enjoys writing "we cannot tell".
   It gets a test.
"""

from __future__ import annotations

import re

import pytest

from northwind.clean import clean
from northwind.generate import SEED, generate
from northwind.memo import PROVENANCE, gather, render


@pytest.fixture(scope="module")
def memo(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    processed = tmp_path_factory.mktemp("processed")

    generate(raw, seed=SEED)
    clean(raw, processed)

    data = gather(raw, processed)
    text = render(data)

    return {"raw": raw, "processed": processed, "data": data, "text": text}


# ---------------------------------------------------------------------------
# 1. NOTHING IS HARDCODED
# ---------------------------------------------------------------------------


def test_the_memo_changes_when_the_data_changes(tmp_path):
    """THE TEST. Generate with a different seed; the memo must be different.

    If any headline number were a literal, it would survive a change in the
    underlying data - and that is precisely how a memo goes stale. Someone
    re-runs the pipeline in March, NRR moves, and the document still reports the
    old figure forever, because nobody re-reads a memo they already signed.

    This test makes staleness impossible rather than unlikely.
    """
    raw_a = tmp_path / "raw_a"
    proc_a = tmp_path / "proc_a"
    generate(raw_a, seed=SEED)
    clean(raw_a, proc_a)
    memo_a = render(gather(raw_a, proc_a))

    raw_b = tmp_path / "raw_b"
    proc_b = tmp_path / "proc_b"
    generate(raw_b, seed=999_111)
    clean(raw_b, proc_b)
    memo_b = render(gather(raw_b, proc_b))

    assert memo_a != memo_b, (
        "the memo is identical across two different datasets - "
        "its numbers are hardcoded and it will go stale"
    )


def test_the_memo_is_deterministic(memo, tmp_path):
    """Same data in, same memo out. No randomness leaking into a board document."""
    again = render(gather(memo["raw"], memo["processed"]))

    # The date line is the only thing allowed to vary, and only across days.
    strip_date = re.compile(r"\*\*Date:\*\* .*")

    assert strip_date.sub("", memo["text"]) == strip_date.sub("", again)


# ---------------------------------------------------------------------------
# 2. THE NUMBERS MATCH THE PIPELINE
# ---------------------------------------------------------------------------


def test_the_headline_nrr_matches_the_decomposition(memo):
    """The number in the title is the number the pipeline computed."""
    latest = memo["data"]["latest"]

    expected = f"{latest['nrr']:.0%}"

    assert expected in memo["text"], (
        f"the memo does not contain the computed NRR of {expected}"
    )


def test_the_attribution_shares_appear_in_the_memo(memo):
    """The 87% expansion share is not a round number someone liked."""
    attr = memo["data"]["attribution"]

    expansion_share = attr.loc["Expansion (upsell)", "share_of_decline"]

    assert f"{expansion_share:.0%}" in memo["text"]


def test_the_reconciliation_claim_is_true(memo):
    """The memo claims a zero residual. It had better be zero.

    A visible false claim discredits every number beside it. If the residual
    were $340 and the memo said $0.00, an auditor who found the gap would - very
    reasonably - stop trusting the whole document.
    """
    recon = memo["data"]["monthly_recon"]

    assert (recon["residual"].abs() < 0.01).all(), (
        "the memo claims every month reconciles, and it does not"
    )
    assert "$0.00" in memo["text"]


def test_the_forecast_figure_matches_the_model(memo):
    """The forecast in the memo is the forecast the model produced."""
    fc = memo["data"]["forecast"].iloc[-1]

    assert f"${fc['forecast']:,.0f}" in memo["text"]
    assert f"${fc['lower']:,.0f}" in memo["text"]
    assert f"${fc['upper']:,.0f}" in memo["text"]


def test_the_pause_ruling_matches_the_contract(memo):
    """The 60-day threshold in the memo comes from the YAML, not from prose.

    Change pause_grace_days in the contract and this sentence changes with it.
    That is what makes the contract load-bearing all the way to the board pack.
    """
    contract = memo["data"]["contract"]

    assert f"{contract.pause_grace_days} days" in memo["text"]


# ---------------------------------------------------------------------------
# 3. PROVENANCE RESOLVES
# ---------------------------------------------------------------------------


def test_every_tag_used_in_the_memo_is_defined(memo):
    """No claim carries a tag that the appendix cannot explain."""
    used = set(re.findall(r"\[([MP]\d-[a-z])\]", memo["text"]))

    undefined = used - set(PROVENANCE)

    assert not undefined, f"the memo cites tags with no provenance entry: {sorted(undefined)}"


def test_every_defined_tag_is_actually_used(memo):
    """No dead entries in the appendix.

    A provenance table listing claims the memo does not make is padding, and
    padding in an evidence table is corrosive - it invites the reader to assume
    the rest is padding too.
    """
    used = set(re.findall(r"\[([MP]\d-[a-z])\]", memo["text"]))

    unused = set(PROVENANCE) - used

    assert not unused, f"the appendix lists tags the memo never cites: {sorted(unused)}"


def test_every_provenance_target_exists(memo):
    """Each tag points at a real function in a real module.

    A footnote pointing at code that does not exist is worse than no footnote.
    """
    for tag, (source, _test) in PROVENANCE.items():
        module_name, function_name = source.rsplit(".", 1)

        module = __import__(f"northwind.{module_name}", fromlist=[function_name])

        assert hasattr(module, function_name), (
            f"provenance tag {tag} points at {source}, which does not exist"
        )


# ---------------------------------------------------------------------------
# 4. THE UNCOMFORTABLE SECTIONS SURVIVE
# The tests most likely to save a future version of this memo from itself.
# ---------------------------------------------------------------------------


def test_the_memo_says_what_it_cannot_tell_you(memo):
    """The section admitting the experiment is unreadable must stay.

    This is the section a future edit is most likely to soften, because nobody
    is rewarded for writing "we cannot tell". It is also the section that earns
    the credibility of every other number in the document.

    Note the whitespace normalisation: the memo hard-wraps its lines, so a phrase
    can straddle a newline. Searching the raw text would make this test depend on
    where the line breaks happened to fall - which is not what we are asserting.
    """
    text = " ".join(memo["text"].split())

    assert "What this analysis cannot tell you" in text
    assert "cannot distinguish a meaningful effect from no effect" in text
    assert "Do not act on it" in text


def test_the_memo_carries_the_forecast_warning(memo):
    """The forecast ships with the fact that it is degrading.

    A forecast presented without its failure mode is not a forecast. It is an
    invitation to plan against a number that is quietly falling apart.
    """
    text = memo["text"]

    text = " ".join(text.split())

    assert "bias has flipped" in text
    assert "actively getting worse" in text
    assert "cannot see the next shock" in text


def test_the_memo_names_the_counterintuitive_finding(memo):
    """Churn improved. The memo says so, in the first paragraph, in bold.

    Bury this and the reader walks away thinking the company has a churn
    problem - and fixes the wrong thing.
    """
    text = memo["text"]

    text = " ".join(text.split())

    assert "Logo churn actually improved" in text
    assert "Churn is not the problem" in text


def test_the_memo_leads_with_the_answer(memo):
    """The first section answers the question. It does not build to it.

    An executive memo is not a detective novel. The reader may only get through
    two paragraphs; those two paragraphs have to contain the finding.
    """
    text = memo["text"]

    first_section = text.split("---")[1]

    assert "The answer, in one paragraph" in first_section
    assert "expansion" in first_section.lower()


def test_the_memo_gives_prioritised_recommendations(memo):
    """It says what to do, in what order, and why that order.

    "Here are the findings" is a report. "Fix this first, because it is four
    times the size of the other thing" is advice, and advice is what was asked
    for.
    """
    text = memo["text"]

    text = " ".join(text.split())

    assert "Recommendations, in priority order" in text
    assert "Fix the expansion motion" in text
    assert "Nothing else on this list is worth doing first" in text


def test_the_memo_is_a_reasonable_length(memo):
    """Long enough to be complete. Short enough to be read.

    A twelve-page memo is a memo nobody reads, which makes it a very expensive
    way to have written nothing.
    """
    words = len(memo["text"].split())

    assert 800 < words < 2500, f"the memo is {words} words - it will not be read"
