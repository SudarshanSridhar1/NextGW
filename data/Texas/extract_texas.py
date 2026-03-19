"""
Extract the Texas (ERCOT) subsystem from case.mat as a standalone MATPOWER mpc.

Texas zone IDs: 301-308
The Texas interconnect is isolated from Eastern/Western via AC lines (0 boundary branches).
Two DC tie lines connect Texas to Eastern — these are dropped and the tied buses
are converted to slack generators (PV buses) to preserve boundary injections.
"""

import scipy.io
import numpy as np
import os

CASE_PATH = os.path.join(os.path.dirname(__file__), "case.mat")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "case_texas.mat")

TEXAS_ZONES = set(range(301, 309))

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
mat = scipy.io.loadmat(CASE_PATH)
mpc_raw = mat["mpc"][0, 0]

bus    = mpc_raw["bus"].copy()       # (N, 17)
gen    = mpc_raw["gen"].copy()       # (G, 25)
branch = mpc_raw["branch"].copy()   # (B, 21)
gencost= mpc_raw["gencost"].copy()  # (G, ?)
dcline = mpc_raw["dcline"].copy()   # (D, 23)
baseMVA= float(mpc_raw["baseMVA"])

# MATPOWER column indices (0-based)
BUS_I   = 0   # bus number
BUS_TYPE= 1   # 1=PQ, 2=PV, 3=ref, 4=isolated
PD      = 2
QD      = 3
BUS_ZONE= 6   # zone_id column

GEN_BUS = 0
GEN_PG  = 1
GEN_PMAX= 8

BR_F    = 0   # from bus
BR_T    = 1   # to bus

DC_F    = 0
DC_T    = 1

# ---------------------------------------------------------------------------
# Texas bus mask
# ---------------------------------------------------------------------------
tx_bus_mask = np.isin(bus[:, BUS_ZONE], list(TEXAS_ZONES))
tx_bus_ids  = set(bus[tx_bus_mask, BUS_I].astype(int))

print(f"Texas buses : {len(tx_bus_ids)}")

# ---------------------------------------------------------------------------
# Generators connected to Texas buses
# ---------------------------------------------------------------------------
gen_mask  = np.array([int(g[GEN_BUS]) in tx_bus_ids for g in gen])
gen_tx    = gen[gen_mask]
gencost_tx= gencost[gen_mask]
print(f"Texas generators : {len(gen_tx)}")

# ---------------------------------------------------------------------------
# Internal AC branches (both endpoints in Texas)
# ---------------------------------------------------------------------------
br_mask = np.array(
    [int(r[BR_F]) in tx_bus_ids and int(r[BR_T]) in tx_bus_ids for r in branch]
)
branch_tx = branch[br_mask]
print(f"Texas internal branches : {len(branch_tx)}")

# ---------------------------------------------------------------------------
# DC lines — find ties touching Texas
# ---------------------------------------------------------------------------
dc_internal = []
dc_ties     = []
for row in dcline:
    f_in = int(row[DC_F]) in tx_bus_ids
    t_in = int(row[DC_T]) in tx_bus_ids
    if f_in and t_in:
        dc_internal.append(row)
    elif f_in or t_in:
        dc_ties.append(row)

dc_internal = np.array(dc_internal) if dc_internal else np.zeros((0, dcline.shape[1]))
print(f"Texas internal DC lines : {len(dc_internal)}")
print(f"DC tie lines (dropped)  : {len(dc_ties)}")

# For each dropped DC tie, inject the scheduled power as a generator on the
# Texas-side bus so power balance is preserved.
extra_gens = []
for row in dc_ties:
    f_in = int(row[DC_F]) in tx_bus_ids
    tx_bus = int(row[DC_F]) if f_in else int(row[DC_T])
    # DC line columns: 0=fbus, 1=tbus, 2=status, 3=Pf (MW injected at from-end)
    pf = float(row[3])   # positive = from→to
    p_inject = pf if f_in else -pf   # power into Texas bus
    # Build a minimal gen row (25 cols) matching MATPOWER gen format
    g = np.zeros(25)
    g[GEN_BUS]  = tx_bus
    g[GEN_PG]   = p_inject
    g[2]        = 0.0    # Qg
    g[3]        = 9999   # Qmax
    g[4]        = -9999  # Qmin
    g[5]        = 1.0    # Vg
    g[6]        = baseMVA
    g[7]        = 1      # status
    g[GEN_PMAX] = max(p_inject, 0) + 1.0
    g[9]        = min(p_inject, 0) - 1.0   # Pmin
    extra_gens.append(g)
    print(f"  DC tie bus {tx_bus}: injecting {p_inject:.1f} MW as fixed generator")

if extra_gens:
    extra_gens   = np.array(extra_gens)
    extra_gcost  = np.zeros((len(extra_gens), gencost_tx.shape[1]))
    # piecewise linear with zero cost
    extra_gcost[:, 0] = 1   # model = 1 (piecewise linear)
    extra_gcost[:, 1] = 2   # 2 points
    gen_tx    = np.vstack([gen_tx, extra_gens])
    gencost_tx= np.vstack([gencost_tx, extra_gcost])

# ---------------------------------------------------------------------------
# Bus subset
# ---------------------------------------------------------------------------
bus_tx = bus[tx_bus_mask].copy()

# Ensure exactly one slack bus (type 3); keep the first ref bus or promote one
ref_mask = bus_tx[:, BUS_TYPE] == 3
if ref_mask.sum() == 0:
    print("No slack bus in Texas subset — promoting bus with highest Pmax gen to ref")
    if len(gen_tx) > 0:
        top_gen_bus = int(gen_tx[np.argmax(gen_tx[:, GEN_PMAX]), GEN_BUS])
        idx = np.where(bus_tx[:, BUS_I] == top_gen_bus)[0]
        if len(idx):
            bus_tx[idx[0], BUS_TYPE] = 3
    else:
        bus_tx[0, BUS_TYPE] = 3
else:
    print(f"Slack buses found: {int(ref_mask.sum())}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
mpc_out = {
    "version": "2",
    "baseMVA": np.array([[baseMVA]]),
    "bus":     bus_tx,
    "gen":     gen_tx,
    "branch":  branch_tx,
    "gencost": gencost_tx,
    "dcline":  dc_internal if len(dc_internal) else np.zeros((0, dcline.shape[1])),
}

scipy.io.savemat(OUT_PATH, {"mpc": mpc_out})
print(f"\nSaved Texas mpc → {OUT_PATH}")
print(f"  buses   : {len(bus_tx)}")
print(f"  gens    : {len(gen_tx)}")
print(f"  branches: {len(branch_tx)}")
