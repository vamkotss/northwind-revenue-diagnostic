"""Seeded mess generator for Northwind Metrics.

WHAT THIS DOES
--------------
Fabricates the complete data history of a fictional B2B SaaS company, then
deliberately corrupts it in eight ways that real billing data is corrupted.

WHY IT EXISTS
-------------
Anyone can analyse clean data. The credibility of this project comes from the
fact that the defects are *known*: we injected them, we can count them, and we
can prove our cleaning layer catches them. That is what a tested defect
generator buys you.

SEEDED
------
All randomness flows from a single seed. Run this twice and you get identical
bytes. Without that, nothing downstream is reproducible and CI is meaningless.

THE BUSINESS STORY (planted deliberately, to be discovered later)
----------------------------------------------------------------
Net revenue retention falls from ~108% to ~94% between Q4 2025 and Q2 2026.
Three causes are planted in the data:

  1. CONTRACTION. From Oct 2025 a competitor undercuts us. Growth-tier
     customers in the SMB segment start downgrading to Starter.
  2. CHURN. A Sept 2025 price rise on usage add-ons pushes heavy-usage
     Starter customers to cancel outright.
  3. EXPANSION COLLAPSE. A Q3 2025 sales reorg kills Enterprise seat
     expansion, which had been carrying NRR.

The punchline the analyst must find: most of the damage is CONTRACTION and
LOST EXPANSION, not logo churn. That is exactly why Sales (who count logos)
and Finance (who count dollars) report different numbers and neither is lying.

TABLES PRODUCED
---------------
  customers               one row per company
  subscriptions           one row per plan period per customer
  invoices                one row per billing document
  usage_events            one row per customer per active day
  experiment_assignments  the contaminated pricing A/B test

Run:  python -m northwind.generate
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURATION
# Every knob lives here so the whole simulation is auditable at a glance.
# ----------------------------------------------------------------------------

SEED = 20260713                  # fixed seed => identical data every single run
N_CUSTOMERS = 5_000

START_DATE = date(2024, 1, 1)    # company history begins
END_DATE = date(2026, 6, 30)     # "today" from the CFO's point of view

# Plan tiers and their list-price monthly recurring revenue, in dollars.
PLANS = {
    "Starter": 99.0,
    "Growth": 499.0,
    "Enterprise": 2400.0,
}

# How new customers distribute across tiers when they sign up.
PLAN_MIX = [0.55, 0.35, 0.10]    # Starter, Growth, Enterprise

SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
INDUSTRIES = ["FinServ", "Healthcare", "Retail", "Logistics", "Manufacturing", "SaaS"]
REGIONS = ["West", "Southwest", "Midwest", "Northeast", "Southeast"]
CHANNELS = ["Inbound", "Outbound", "Partner", "Paid Search", "Referral"]

# The three planted shocks. Dates the analyst will eventually have to find.
PRICE_RISE_DATE = date(2025, 9, 1)      # usage add-on rates increased
COMPETITOR_DATE = date(2025, 10, 1)     # competitor launches; downgrades begin
SALES_REORG_DATE = date(2025, 7, 1)     # expansion motion breaks

# ---------------------------------------------------------------------------
# PAUSE BEHAVIOUR
#
# This is the crux of the whole project, so it is modelled honestly rather than
# asserted. Customers pause for a while and then either come back or do not.
#
# The KEY property: the longer someone stays paused, the less likely they ever
# return. That decay is what makes an empirical churn threshold derivable. The
# analyst is meant to plot return rate against pause length, find where the
# curve flattens, and cut there - instead of picking a round number because it
# sounds sensible.
#
# With the numbers below, roughly 88% of all returns happen within 60 days, and
# beyond that the monthly return rate collapses under 2%. That is the finding.
# It is planted, but it is not given away: nothing in the data says "60 days".
# ---------------------------------------------------------------------------

# Monthly probability that an active customer pauses rather than churning.
PAUSE_HAZARD = {"Starter": 0.010, "Growth": 0.008, "Enterprise": 0.005}

# How long pauses last, in months, and how likely a return is at each length.
PAUSE_MONTHS = [1, 2, 3, 4, 5, 6]
PAUSE_MONTH_WEIGHTS = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
RESURRECTION_PROB = {1: 0.80, 2: 0.62, 3: 0.28, 4: 0.12, 5: 0.06, 6: 0.03}

# Defect injection rates. These are the numbers the tests assert against.
DEFECT_RATES = {
    "duplicate_invoices": 0.006,       # 0.6% of invoices get a twin
    "timezone_drift": 0.030,           # 3% of usage rows carry a bad offset
    "retroactive_subscription": 0.015,  # 1.5% of subs get a backdated correction
    "missing_values": 0.020,           # 2% nulls in customer attributes
    "dirty_categoricals": 0.040,       # 4% of category labels are malformed
    "dirty_amounts": 0.050,            # 5% of invoice amounts are strings
    "orphan_invoices": 0.003,          # 0.3% reference a nonexistent customer
    "ambiguous_pause": 0.025,          # 2.5% of subs sit in a "paused" limbo
}


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------


def month_starts(first: date, last: date) -> list[date]:
    """Return the first day of every month between two dates, inclusive.

    Used to bill customers monthly. We walk month by month rather than adding
    30 days, because months are not 30 days and billing cares about that.
    """
    out = []
    y, m = first.year, first.month
    while date(y, m, 1) <= last:
        out.append(date(y, m, 1))
        # Advance one month, rolling the year over in December.
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def add_months(d: date, n: int) -> date:
    """Add n months to a date, clamping the day so we never build Feb 31st."""
    total = d.month - 1 + n
    year = d.year + total // 12
    month = total % 12 + 1
    # Days in the target month: step to the 1st of the NEXT month, back off one.
    if month == 12:
        last_day = 31
    else:
        last_day = (date(year, month + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(d.day, last_day))


# ----------------------------------------------------------------------------
# STAGE 1 - CUSTOMERS
# ----------------------------------------------------------------------------


def build_customers(rng: np.random.Generator) -> pd.DataFrame:
    """One row per company that ever signed up."""
    n = N_CUSTOMERS

    # Signup dates are spread across the history, weighted toward later months
    # because the company was growing. We draw a day offset from a distribution
    # that leans right (more recent signups than old ones).
    span_days = (END_DATE - START_DATE).days
    # Beta(2, 1.4) leans toward 1.0, i.e. toward the recent end of the window.
    offsets = (rng.beta(2.0, 1.4, size=n) * span_days).astype(int)
    signup_dates = [START_DATE + timedelta(days=int(o)) for o in offsets]

    df = pd.DataFrame(
        {
            "customer_id": [f"CUS-{i:05d}" for i in range(1, n + 1)],
            "company_name": [f"Company {i:05d}" for i in range(1, n + 1)],
            "signup_date": signup_dates,
            "segment": rng.choice(SEGMENTS, size=n, p=[0.60, 0.30, 0.10]),
            "industry": rng.choice(INDUSTRIES, size=n),
            "region": rng.choice(REGIONS, size=n),
            "acquisition_channel": rng.choice(CHANNELS, size=n, p=[0.30, 0.20, 0.15, 0.25, 0.10]),
        }
    )
    return df


# ----------------------------------------------------------------------------
# STAGE 2 - SUBSCRIPTIONS
# This is where the business story gets planted.
# ----------------------------------------------------------------------------


def _churn_hazard(current: date, plan: str, segment: str, heavy_user: bool) -> float:
    """Monthly probability that a customer cancels outright (logo churn).

    Baseline churn is low. The Sept 2025 price rise sharply raises churn for
    heavy-usage Starter customers - that is planted cause #2.
    """
    base = {"Starter": 0.022, "Growth": 0.012, "Enterprise": 0.004}[plan]

    if current >= PRICE_RISE_DATE and plan == "Starter" and heavy_user:
        base *= 2.4          # the price rise drove them out

    if segment == "SMB":
        base *= 1.25         # small companies always churn more
    return min(base, 0.35)


def _downgrade_hazard(current: date, plan: str, segment: str) -> float:
    """Monthly probability of moving DOWN a tier (contraction).

    Near zero until the competitor launches, then Growth/SMB bleeds badly.
    That is planted cause #1 - and it is the biggest single driver of the
    NRR decline, while being completely invisible to anyone counting logos.
    """
    if current < COMPETITOR_DATE:
        return 0.003
    if plan == "Growth" and segment == "SMB":
        return 0.038         # the competitor's sweet spot
    if plan == "Growth":
        return 0.018
    if plan == "Enterprise":
        return 0.008
    return 0.0               # Starter cannot go lower


def _expansion_hazard(current: date, plan: str) -> float:
    """Monthly probability of expanding (upgrading tier or adding seats).

    Healthy until the Q3 2025 sales reorg, then Enterprise expansion collapses.
    That is planted cause #3 - the quiet one. Expansion revenue does not
    disappear from a churn report, it just stops showing up, and NRR falls.
    """
    if plan == "Enterprise":
        return 0.052 if current < SALES_REORG_DATE else 0.011
    if plan == "Growth":
        return 0.040 if current < SALES_REORG_DATE else 0.015
    return 0.024             # Starter -> Growth, largely self-serve, unaffected


def build_subscriptions(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Walk each customer month by month and record their plan history.

    Every customer produces one or more subscription rows. A new row is opened
    whenever the plan or price changes; the previous row is closed out.
    """
    rows = []
    sub_counter = 0

    # Pre-draw which customers are heavy users. Heavy users pay usage add-ons,
    # so the Sept 2025 price rise hits them and only them.
    heavy = rng.random(len(customers)) < 0.28

    for idx, cust in enumerate(customers.itertuples(index=False)):
        # Starting plan, drawn from the overall mix.
        plan = rng.choice(list(PLANS.keys()), p=PLAN_MIX)

        # Enterprise-segment companies rarely start on Starter.
        if cust.segment == "Enterprise" and plan == "Starter":
            plan = "Growth"

        period_start = cust.signup_date
        # Seat count drives MRR above the list price for bigger accounts.
        seats = int(rng.integers(1, 6) if plan == "Starter" else
                    rng.integers(5, 30) if plan == "Growth" else
                    rng.integers(30, 200))

        # Walk forward month by month from signup until they churn or we hit today.
        current = cust.signup_date
        alive = True

        while alive and current < END_DATE:
            next_month = add_months(current, 1)

            # Roll the dice on the three life events, in priority order.
            r = rng.random()

            churned = r < _churn_hazard(current, plan, cust.segment, bool(heavy[idx]))
            downgraded = (not churned) and (
                rng.random() < _downgrade_hazard(current, plan, cust.segment)
            )
            expanded = (not churned and not downgraded) and (
                rng.random() < _expansion_hazard(current, plan)
            )
            paused = (not churned and not downgraded and not expanded) and (
                rng.random() < PAUSE_HAZARD[plan]
            )

            if churned:
                sub_counter += 1
                rows.append(
                    {
                        "subscription_id": f"SUB-{sub_counter:06d}",
                        "customer_id": cust.customer_id,
                        "plan_tier": plan,
                        "seats": seats,
                        "mrr": round(PLANS[plan] * (1 + 0.08 * (seats - 1)), 2),
                        "period_start": period_start,
                        "period_end": next_month,
                        "status": "churned",
                        "is_correction": False,
                    }
                )
                alive = False

            elif downgraded:
                # Close the current row, open a cheaper one next month.
                sub_counter += 1
                rows.append(
                    {
                        "subscription_id": f"SUB-{sub_counter:06d}",
                        "customer_id": cust.customer_id,
                        "plan_tier": plan,
                        "seats": seats,
                        "mrr": round(PLANS[plan] * (1 + 0.08 * (seats - 1)), 2),
                        "period_start": period_start,
                        "period_end": next_month,
                        "status": "downgraded",
                        "is_correction": False,
                    }
                )
                plan = "Growth" if plan == "Enterprise" else "Starter"
                seats = max(1, int(seats * 0.5))
                period_start = next_month

            elif paused:
                # Close the current period with status 'paused'. Note what the
                # source system does NOT record: whether they are coming back.
                # Nobody knows at the time. That is the whole problem.
                sub_counter += 1
                rows.append(
                    {
                        "subscription_id": f"SUB-{sub_counter:06d}",
                        "customer_id": cust.customer_id,
                        "plan_tier": plan,
                        "seats": seats,
                        "mrr": round(PLANS[plan] * (1 + 0.08 * (seats - 1)), 2),
                        "period_start": period_start,
                        "period_end": next_month,
                        "status": "paused",
                        "is_correction": False,
                    }
                )

                # How long the pause lasts, and whether they ever come back.
                # Return probability DECAYS sharply with duration - that decay
                # is the signal the analyst must find.
                months_paused = int(rng.choice(PAUSE_MONTHS, p=PAUSE_MONTH_WEIGHTS))
                comes_back = rng.random() < RESURRECTION_PROB[months_paused]

                if comes_back:
                    # They resume, usually with fewer seats than before.
                    resume = add_months(next_month, months_paused)
                    if resume >= END_DATE:
                        alive = False          # pause outlasts our window
                    else:
                        seats = max(1, int(seats * rng.uniform(0.6, 1.0)))
                        period_start = resume
                        current = resume
                        continue               # skip the normal month advance
                else:
                    # They never return. The source system never marks them
                    # cancelled - the row just sits there, paused, forever.
                    alive = False

            elif expanded:
                sub_counter += 1
                rows.append(
                    {
                        "subscription_id": f"SUB-{sub_counter:06d}",
                        "customer_id": cust.customer_id,
                        "plan_tier": plan,
                        "seats": seats,
                        "mrr": round(PLANS[plan] * (1 + 0.08 * (seats - 1)), 2),
                        "period_start": period_start,
                        "period_end": next_month,
                        "status": "expanded",
                        "is_correction": False,
                    }
                )
                # Expansion is either a tier upgrade or a seat increase.
                if plan == "Starter" and rng.random() < 0.4:
                    plan = "Growth"
                    seats = max(5, seats)
                elif plan == "Growth" and rng.random() < 0.25:
                    plan = "Enterprise"
                    seats = max(30, seats)
                else:
                    seats = int(seats * rng.uniform(1.15, 1.6)) + 1
                period_start = next_month

            current = next_month

        # Still alive at END_DATE: leave the final period open.
        if alive:
            sub_counter += 1
            rows.append(
                {
                    "subscription_id": f"SUB-{sub_counter:06d}",
                    "customer_id": cust.customer_id,
                    "plan_tier": plan,
                    "seats": seats,
                    "mrr": round(PLANS[plan] * (1 + 0.08 * (seats - 1)), 2),
                    "period_start": period_start,
                    "period_end": None,          # open-ended = currently active
                    "status": "active",
                    "is_correction": False,
                }
            )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# STAGE 3 - INVOICES
# The ledger. This is what Finance sees, and it will NOT tie to subscriptions.
# ----------------------------------------------------------------------------


def build_invoices(subs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One invoice per customer per billing month, plus usage add-on charges."""
    rows = []
    inv_counter = 0

    for sub in subs.itertuples(index=False):
        start = sub.period_start
        end = sub.period_end if sub.period_end is not None else END_DATE

        for bill_date in month_starts(start, end):
            if bill_date < start or bill_date > end:
                continue

            inv_counter += 1

            # Base charge is the subscription MRR.
            base = sub.mrr

            # Usage add-ons: overage charges, higher after the Sept 2025 rise.
            multiplier = 1.45 if bill_date >= PRICE_RISE_DATE else 1.0
            addon = 0.0
            if rng.random() < 0.42:
                addon = round(float(rng.gamma(1.4, 22.0)) * multiplier, 2)

            amount = round(base + addon, 2)

            # Payment behaviour: most pay, some are late, a few never do.
            roll = rng.random()
            if roll < 0.90:
                status = "paid"
                paid_date = bill_date + timedelta(days=int(rng.integers(1, 20)))
            elif roll < 0.97:
                status = "late"
                paid_date = bill_date + timedelta(days=int(rng.integers(35, 95)))
            else:
                status = "unpaid"
                paid_date = None

            rows.append(
                {
                    "invoice_id": f"INV-{inv_counter:07d}",
                    "customer_id": sub.customer_id,
                    "subscription_id": sub.subscription_id,
                    "issued_date": bill_date,
                    "amount": amount,
                    "base_amount": base,
                    "addon_amount": addon,
                    "status": status,
                    "paid_date": paid_date,
                }
            )

    df = pd.DataFrame(rows)

    # Every real invoice is its own document; it reverses nothing.
    df["reverses_invoice_id"] = None

    # Refunds: recorded as NEGATIVE invoices, arriving weeks after the original.
    # Finance nets these; analytics teams routinely forget to. Real trap.
    #
    # Crucially, a refund CARRIES A REFERENCE to the invoice it reverses -
    # exactly as a real credit note does. Without that reference, ruling R5
    # ("attribute the refund to the ORIGINAL invoice month") would be
    # unimplementable, and we would be shipping a rule the code cannot honour.
    n_refunds = int(len(df) * 0.012)
    refund_src = df.sample(n=n_refunds, random_state=int(rng.integers(0, 10**6)))
    refunds = refund_src.copy()
    refunds["reverses_invoice_id"] = refund_src["invoice_id"].to_numpy()
    refunds["invoice_id"] = [f"INV-R{i:06d}" for i in range(1, n_refunds + 1)]
    refunds["amount"] = -refunds["amount"].abs()
    refunds["base_amount"] = -refunds["base_amount"].abs()
    refunds["addon_amount"] = -refunds["addon_amount"].abs()
    refunds["status"] = "refunded"
    refunds["issued_date"] = [
        d + timedelta(days=int(rng.integers(10, 60))) for d in refunds["issued_date"]
    ]

    return pd.concat([df, refunds], ignore_index=True)


# ----------------------------------------------------------------------------
# STAGE 4 - USAGE EVENTS
# The biggest table. One row per customer per active day.
# ----------------------------------------------------------------------------


def build_usage(subs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Daily product usage for every customer, for every day they were active.

    Built vectorised, because a Python loop over ~2M rows is unbearably slow.
    """
    frames = []

    for sub in subs.itertuples(index=False):
        start = sub.period_start
        end = sub.period_end if sub.period_end is not None else END_DATE
        n_days = (end - start).days
        if n_days <= 0:
            continue

        days = pd.date_range(start=start, periods=n_days, freq="D")

        # Active seats wobble around the contracted seat count.
        active_seats = np.clip(
            rng.normal(sub.seats * 0.72, sub.seats * 0.18, size=n_days), 0, sub.seats
        ).round().astype(int)

        # API calls scale with seats, with weekday/weekend seasonality.
        weekday_factor = np.where(days.dayofweek < 5, 1.0, 0.35)
        api_calls = (
            rng.gamma(2.0, 40.0, size=n_days) * sub.seats * weekday_factor
        ).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "customer_id": sub.customer_id,
                    "event_date": days,
                    "active_seats": active_seats,
                    "api_calls": api_calls,
                }
            )
        )

    usage = pd.concat(frames, ignore_index=True)

    # Give every row a timestamp, not just a date. Timezone bugs need a clock.
    hours = rng.integers(0, 24, size=len(usage))
    usage["event_ts"] = usage["event_date"] + pd.to_timedelta(hours, unit="h")

    return usage


# ----------------------------------------------------------------------------
# STAGE 5 - THE CONTAMINATED A/B TEST
# ----------------------------------------------------------------------------


def build_experiment(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The Q1 2026 pricing experiment - broken in two ways, on purpose.

    THE EXPERIMENT
    A retention offer: treatment customers were shown a discounted renewal.
    The question the VP wants answered is "did it reduce churn?"

    DEFECT A: SAMPLE RATIO MISMATCH, WITH A CAUSE.
    The split was meant to be 50/50 and lands near 55/45. But an SRM is a
    SYMPTOM, not a disease. The disease is that the assignment code bucketed on
    a field that correlates with customer size - so SMB customers were pushed
    disproportionately into control, and Enterprise customers into treatment.

    This is what makes an SRM lethal rather than merely untidy. The two arms are
    no longer comparable. Enterprise customers churn far less than SMB ones - so
    treatment will LOOK like it worked, brilliantly, even though the true effect
    is zero. An analyst who checks the p-value and not the split will ship a
    finding that is exactly backwards.

    DEFECT B: CONTAMINATION.
    A deploy mid-experiment re-randomised ~4% of customers, so they appear in
    BOTH arms. Their outcomes are unusable and must be excluded before any
    readout.

    THE TRUE TREATMENT EFFECT IS ZERO. The discount did nothing. Everything the
    naive analysis will find is selection bias.
    """
    exp_start = date(2026, 1, 15)

    eligible = customers[customers["signup_date"] < exp_start].copy()
    eligible = eligible.sample(frac=0.45, random_state=int(rng.integers(0, 10**6)))

    n = len(eligible)

    # DEFECT A: the assignment probability DEPENDS ON SEGMENT. This is the bug.
    # A correct experiment would use a constant 0.50 here for everyone.
    control_prob = {
        "SMB": 0.62,          # small customers pushed toward control
        "Mid-Market": 0.50,
        "Enterprise": 0.35,   # large customers pushed toward treatment
    }

    arms = []
    for segment in eligible["segment"]:
        p_control = control_prob.get(segment, 0.50)
        arms.append("control" if rng.random() < p_control else "treatment")

    df = pd.DataFrame(
        {
            "customer_id": eligible["customer_id"].to_numpy(),
            "variant": arms,
            "assigned_date": exp_start,
        }
    )

    # DEFECT B: 4% of customers get a SECOND assignment in the opposite arm.
    n_contaminated = int(n * 0.04)
    contaminated = df.sample(n=n_contaminated, random_state=int(rng.integers(0, 10**6))).copy()
    contaminated["variant"] = np.where(
        contaminated["variant"] == "control", "treatment", "control"
    )
    contaminated["assigned_date"] = exp_start + timedelta(days=9)

    return pd.concat([df, contaminated], ignore_index=True)


# ----------------------------------------------------------------------------
# STAGE 6 - THE MESS
# Everything above is clean. Now we break it, on purpose, in eight ways.
# ----------------------------------------------------------------------------


def inject_defects(
    customers: pd.DataFrame,
    subs: pd.DataFrame,
    invoices: pd.DataFrame,
    usage: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply all eight defect types and return a manifest of what we did.

    The manifest matters as much as the mess. It is what the tests assert
    against, and it is what lets you say in an interview: "I know there were
    exactly 512 duplicate invoices, because I put them there."
    """
    manifest = []

    def log(defect: str, table: str, count: int, note: str) -> None:
        manifest.append(
            {"defect": defect, "table": table, "rows_affected": count, "note": note}
        )

    # --- DEFECT 1: duplicate invoices -------------------------------------
    # Two flavours: exact twins (same ID, double-loaded from the billing export)
    # and near-twins (new ID, same everything else - a retry that both landed).
    n_dupes = int(len(invoices) * DEFECT_RATES["duplicate_invoices"])
    dupes = invoices.sample(n=n_dupes, random_state=int(rng.integers(0, 10**6))).copy()
    half = n_dupes // 2
    # Near-twins get a fresh invoice_id, so a naive "drop_duplicates" misses them.
    dupes.iloc[:half, dupes.columns.get_loc("invoice_id")] = [
        f"INV-D{i:06d}" for i in range(1, half + 1)
    ]
    invoices = pd.concat([invoices, dupes], ignore_index=True)
    log("duplicate_invoices", "invoices", n_dupes,
        f"{half} near-duplicates with new IDs, {n_dupes - half} exact duplicate IDs")

    # --- DEFECT 2: timezone drift -----------------------------------------
    # Some usage rows were logged in US/Central and written as if UTC. Events
    # near midnight land on the WRONG DAY, which quietly corrupts daily metrics.
    n_tz = int(len(usage) * DEFECT_RATES["timezone_drift"])
    tz_idx = rng.choice(usage.index, size=n_tz, replace=False)
    usage.loc[tz_idx, "event_ts"] = usage.loc[tz_idx, "event_ts"] - pd.Timedelta(hours=6)
    log("timezone_drift", "usage_events", n_tz,
        "6-hour negative offset; events near midnight fall on the previous day")

    # --- DEFECT 3: retroactive subscription corrections --------------------
    # Finance backdated some plan changes. The same customer-month now appears
    # twice with different MRR. Sum naively and you double-count revenue.
    n_retro = int(len(subs) * DEFECT_RATES["retroactive_subscription"])
    retro = subs.sample(n=n_retro, random_state=int(rng.integers(0, 10**6))).copy()
    retro["subscription_id"] = [f"SUB-C{i:06d}" for i in range(1, n_retro + 1)]
    retro["mrr"] = (retro["mrr"] * rng.uniform(0.75, 0.95, size=n_retro)).round(2)
    retro["is_correction"] = True
    subs = pd.concat([subs, retro], ignore_index=True)
    log("retroactive_subscription", "subscriptions", n_retro,
        "backdated corrections overlapping existing periods; is_correction=True")

    # --- DEFECT 4: missing values -----------------------------------------
    n_missing = int(len(customers) * DEFECT_RATES["missing_values"])
    for col in ["segment", "industry", "region"]:
        idx = rng.choice(customers.index, size=n_missing, replace=False)
        customers.loc[idx, col] = None
    log("missing_values", "customers", n_missing * 3,
        "nulls scattered across segment, industry, region")

    # --- DEFECT 5: dirty categoricals -------------------------------------
    # The same value spelled five ways, because five systems wrote to this field.
    n_dirty = int(len(customers) * DEFECT_RATES["dirty_categoricals"])
    idx = rng.choice(
        customers.dropna(subset=["segment"]).index, size=n_dirty, replace=False
    )
    variants = {
        "SMB": ["smb", "S.M.B.", " SMB ", "Smb"],
        "Mid-Market": ["mid-market", "MidMarket", "Mid Market", " mid-market"],
        "Enterprise": ["enterprise", "ENT", "Enterprise ", "ENTERPRISE"],
    }
    for i in idx:
        clean = customers.at[i, "segment"]
        if clean in variants:
            customers.at[i, "segment"] = rng.choice(variants[clean])
    log("dirty_categoricals", "customers", n_dirty,
        "segment spelled inconsistently: casing, spacing, abbreviations")

    # --- DEFECT 6: dirty amounts ------------------------------------------
    # Amounts exported as text: currency symbols, thousands separators, and
    # accounting-style negatives in parentheses. Reads as a string column.
    invoices["amount"] = invoices["amount"].astype(object)
    n_amt = int(len(invoices) * DEFECT_RATES["dirty_amounts"])
    amt_idx = rng.choice(invoices.index, size=n_amt, replace=False)
    for i in amt_idx:
        val = invoices.at[i, "amount"]
        if val < 0:
            invoices.at[i, "amount"] = f"({abs(val):,.2f})"   # accounting negative
        else:
            invoices.at[i, "amount"] = f"${val:,.2f}"          # $1,234.56
    log("dirty_amounts", "invoices", n_amt,
        "amounts as strings: '$1,234.56' and accounting negatives '(500.00)'")

    # --- DEFECT 7: orphan invoices ----------------------------------------
    # Invoices pointing at customer IDs that do not exist. A left join silently
    # drops them and your revenue total is quietly wrong.
    n_orphan = int(len(invoices) * DEFECT_RATES["orphan_invoices"])
    orphans = invoices.sample(n=n_orphan, random_state=int(rng.integers(0, 10**6))).copy()
    orphans["invoice_id"] = [f"INV-O{i:06d}" for i in range(1, n_orphan + 1)]
    orphans["customer_id"] = [f"CUS-9{i:04d}" for i in range(1, n_orphan + 1)]
    invoices = pd.concat([invoices, orphans], ignore_index=True)
    log("orphan_invoices", "invoices", n_orphan,
        "customer_id values with no matching row in customers")

    # --- DEFECT 8: mislabelled cancellations ------------------------------
    # Real pauses already exist in the data, with real durations and real
    # return behaviour. THIS defect is different: some customers who genuinely
    # CANCELLED were never marked cancelled in the CRM. A rep forgot. So they
    # sit in the system as 'paused' forever.
    #
    # The consequence: 'paused' is now a mixed bag. Some of those customers are
    # coming back. Some left months ago and are never coming back. The source
    # system cannot tell you which, and that ambiguity is precisely why Sales
    # and Finance report different churn numbers - and why the analyst has to
    # rule on it from evidence rather than opinion.
    churned_idx = subs[subs["status"] == "churned"].index
    n_mislabel = int(len(subs) * DEFECT_RATES["ambiguous_pause"])
    n_mislabel = min(n_mislabel, len(churned_idx))
    mislabel_idx = rng.choice(churned_idx, size=n_mislabel, replace=False)
    subs.loc[mislabel_idx, "status"] = "paused"
    log("ambiguous_pause", "subscriptions", n_mislabel,
        "true cancellations never marked cancelled; they sit as 'paused' forever")

    return customers, subs, invoices, usage, pd.DataFrame(manifest)


# ----------------------------------------------------------------------------
# ORCHESTRATION
# ----------------------------------------------------------------------------


def generate(out_dir: Path, seed: int = SEED) -> dict[str, int]:
    """Run every stage and write Parquet files. Returns row counts."""
    rng = np.random.default_rng(seed)

    print(f"Seed: {seed}")
    print("Building customers...")
    customers = build_customers(rng)

    print("Building subscriptions (this walks every customer month by month)...")
    subs = build_subscriptions(customers, rng)

    print("Building invoices...")
    invoices = build_invoices(subs, rng)

    print("Building usage events (the big one)...")
    usage = build_usage(subs, rng)

    print("Building the contaminated experiment...")
    experiment = build_experiment(customers, rng)

    print("Injecting defects...")
    customers, subs, invoices, usage, manifest = inject_defects(
        customers, subs, invoices, usage, rng
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "customers": customers,
        "subscriptions": subs,
        "invoices": invoices,
        "usage_events": usage,
        "experiment_assignments": experiment,
        "defect_manifest": manifest,
    }

    counts = {}
    for name, df in tables.items():
        # Invoices ship as CSV on purpose. That is how billing systems actually
        # export: one flat text file, no types, dates as strings, amounts as
        # strings. Everything else is Parquet, which preserves types properly.
        # The contrast is the point - you will feel the difference immediately.
        if name == "invoices":
            path = out_dir / f"{name}.csv"
            df.to_csv(path, index=False)
        else:
            path = out_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)

        counts[name] = len(df)
        print(f"  {name:24s} {len(df):>9,} rows  ->  {path.name}")

    print("\nDefect manifest:")
    print(manifest.to_string(index=False))

    return counts


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate messy Northwind data.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="Directory to write Parquet files into (default: data/raw)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    args = parser.parse_args()

    started = datetime.now()
    generate(args.out, args.seed)
    elapsed = (datetime.now() - started).total_seconds()
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
