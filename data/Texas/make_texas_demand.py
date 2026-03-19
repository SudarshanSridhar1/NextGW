"""
Build texas_demand.csv: hourly load timeseries for every Texas bus.

Method: each bus gets a fixed fraction of its zone's demand,
        proportional to its base-case Pd (from bus.csv).
        Buses with Pd == 0 are excluded (they carry no load).
"""

import pandas as pd
import numpy as np

BASE = "/Users/nangu729/Documents/data/USATestSystem 2/base_grid"

bus    = pd.read_csv(f"{BASE}/bus.csv")
demand = pd.read_csv(f"{BASE}/demand.csv")

TEXAS_ZONES = list(range(301, 309))

# ── Texas buses with load ──────────────────────────────────────────────────
tx_bus = bus[bus["zone_id"].isin(TEXAS_ZONES)].copy()

# fraction of zone total each bus represents (0 for buses with Pd == 0)
zone_totals = tx_bus.groupby("zone_id")["Pd"].transform("sum")
tx_bus["share"] = tx_bus["Pd"] / zone_totals.replace(0, np.nan)
tx_bus["share"] = tx_bus["share"].fillna(0.0)

# ── Build output ───────────────────────────────────────────────────────────
cols = {"UTC Time": demand["UTC Time"].values}

for zone in TEXAS_ZONES:
    zone_ts = demand[str(zone)].values          # (8784,)
    buses_z = tx_bus[tx_bus["zone_id"] == zone]
    for _, row in buses_z.iterrows():
        cols[int(row["bus_id"])] = zone_ts * row["share"]

# sort bus columns numerically, time first
sorted_bus_ids = sorted(k for k in cols if k != "UTC Time")
out = pd.DataFrame({"UTC Time": cols["UTC Time"],
                    **{b: cols[b] for b in sorted_bus_ids}})

out.to_csv(f"{BASE}/texas_demand.csv", index=False)

print(f"Saved texas_demand.csv")
print(f"  Timesteps : {len(out)}")
print(f"  Load buses: {len(sorted_bus_ids)}")
print(f"  Zones     : {TEXAS_ZONES}")
print(out.iloc[:2, :6])
