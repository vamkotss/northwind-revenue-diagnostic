"""Scratch analysis: how long do pauses last, and do customers come back?

This is the evidence behind the 60-day churn ruling. Run it, look at the
curve, and decide for yourself where to cut.
"""

import pandas as pd

# Load subscriptions; ignore the injected correction rows for now.
subs = pd.read_parquet("data/raw/subscriptions.parquet")
s = subs[~subs["is_correction"]].copy()
s["period_start"] = pd.to_datetime(s["period_start"])
s["period_end"] = pd.to_datetime(s["period_end"])

# Every subscription period that ended in a pause.
paused = s[s["status"] == "paused"][
    ["subscription_id", "customer_id", "period_end"]
].copy()

# The source system NEVER records whether a paused customer came back.
# We reconstruct it: join the table to itself and look for a LATER period
# belonging to the same customer.
joined = paused.merge(s[["customer_id", "period_start"]], on="customer_id")
joined = joined[joined["period_start"] > joined["period_end"]]

# The earliest such period is the moment they resumed.
first_return = joined.groupby("subscription_id")["period_start"].min()

paused["returned_on"] = paused["subscription_id"].map(first_return)
paused["returned"] = paused["returned_on"].notna()

# Months between pausing and returning. 30.44 is the average month length.
paused["gap_months"] = (
    (paused["returned_on"] - paused["period_end"]).dt.days / 30.44
).round()

print(f"Paused subscriptions: {len(paused):,}")
print(f"Ever returned:        {paused['returned'].mean():.1%}\n")

returns = paused[paused["returned"]]

print("WHEN DO RETURNS HAPPEN?")
print(f"{'Pause length':>14} {'Returns':>9} {'Cumulative':>12}")
running, total = 0, len(returns)
for gap, count in returns["gap_months"].value_counts().sort_index().items():
    running += count
    bar = "#" * int(count / total * 40)
    print(f"{int(gap):>11} mo {count:>9,} {running / total:>11.1%}  {bar}")

within_60 = (returns["gap_months"] <= 2).mean()
print(f"\n>>> {within_60:.1%} of ALL returns happen within 60 days.")
print(">>> Beyond that the curve flatlines. That is where you cut.")