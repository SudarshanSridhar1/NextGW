"""
hosting_capacity_parallel.py
─────────────────────────────
Same computation as hosting_capacity.py but runs all per-bus LPs in parallel
using ProcessPoolExecutor.  Large matrices are pushed to each worker once via
the pool initializer, not once per task — so serialisation overhead is minimal.

Expected speed-up: ~N_workers× (Gurobi licence permitting).
"""

import os
import sys
import time
import numpy as np
import scipy.io
import pandas as pd
import cvxpy as cp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

# ── Paths ──────────────────────────────────────────────────────────────────────
TEXAS_DIR   = os.path.dirname(os.path.abspath(__file__))
CASE_PATH   = os.path.join(TEXAS_DIR, "case_texas.mat")
DEMAND_PATH = os.path.join(TEXAS_DIR, "texas_demand.csv.gz")
OUT_PATH    = os.path.join(TEXAS_DIR, "hosting_capacity_results.csv")

sys.path.insert(0, TEXAS_DIR)
from ptdf import compute_ptdf

# ── How many parallel workers to use ──────────────────────────────────────────
# Gurobi academic/commercial licences support concurrent processes.
# Lower this if you hit licence or memory limits.
N_WORKERS = min(os.cpu_count() or 4, 16)


# ── Per-worker global store (populated by initializer) ────────────────────────
_W = {}   # keys: HA_g, H, Hb_all, rhs_all, F_max, G_min, G_max


def _worker_init(HA_g, H, Hb_all, rhs_all, F_max, G_min, G_max):
    """Called once when each worker process starts; stores arrays as globals."""
    _W["HA_g"]    = HA_g
    _W["H"]       = H
    _W["Hb_all"]  = Hb_all
    _W["rhs_all"] = rhs_all
    _W["F_max"]   = F_max
    _W["G_min"]   = G_min
    _W["G_max"]   = G_max


def _solve_bus_worker(k):
    """Thin wrapper that reads worker-local globals and calls the LP solver."""
    return _solve_bus(
        k,
        _W["HA_g"], _W["H"], _W["Hb_all"], _W["rhs_all"],
        _W["F_max"], _W["G_min"], _W["G_max"],
    )


# ── LP solver (unchanged logic from hosting_capacity.py) ──────────────────────
def _solve_bus(k, HA_g, H, Hb_all, rhs_all, F_max, G_min, G_max):
    """Return (bus_index, HC_network_MW) for bus index k."""
    n_gen = len(G_max)
    H_k   = H[:, k]

    g  = cp.Variable(n_gen, name="g")
    dL = cp.Variable(nonneg=True, name="dL")

    Hb_param  = cp.Parameter(len(F_max), name="Hb")
    rhs_param = cp.Parameter(name="rhs")

    flow = HA_g @ g - H_k * dL

    constraints = [
        cp.sum(g) - dL == rhs_param,
        g >= G_min,
        g <= G_max,
        flow >= Hb_param - F_max,
        flow <= Hb_param + F_max,
    ]

    problem = cp.Problem(cp.Maximize(dL), constraints)

    min_hc = np.inf
    for t in range(Hb_all.shape[1]):
        Hb_param.value  = Hb_all[:, t]
        rhs_param.value = float(rhs_all[t])

        try:
            problem.solve(solver=cp.GUROBI, warm_start=True, verbose=False)
        except cp.SolverError:
            return k, 0.0

        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            min_hc = min(min_hc, float(dL.value))
        else:
            return k, 0.0

    return k, (0.0 if min_hc == np.inf else min_hc)


# ── Data loaders (unchanged) ───────────────────────────────────────────────────
def load_network(case_path):
    mat = scipy.io.loadmat(case_path)
    mpc = mat["mpc"][0, 0]

    bus_arr    = mpc["bus"]
    gen_arr    = mpc["gen"]
    branch_arr = mpc["branch"]

    bus_ids = bus_arr[:, 0].astype(int)
    N_BUS   = len(bus_ids)
    bus_idx = {bid: i for i, bid in enumerate(bus_ids)}

    active_gen = gen_arr[:, 7] == 1
    gen_a      = gen_arr[active_gen]
    GEN_BUS    = gen_a[:, 0].astype(int)
    G_max      = gen_a[:, 8].copy()
    G_min      = gen_a[:, 9].copy()

    GEN_IDX = np.array([bus_idx[b] for b in GEN_BUS])
    G_inc   = np.zeros((N_BUS, len(G_max)))
    for j, bi in enumerate(GEN_IDX):
        G_inc[bi, j] = 1.0

    active_br = branch_arr[:, 10] == 1
    F_max     = branch_arr[active_br, 5].copy()

    return bus_ids, N_BUS, G_min, G_max, G_inc, F_max


def load_demand(demand_path, bus_ids):
    df = pd.read_csv(demand_path, index_col=0, parse_dates=True)
    df = df[[str(b) for b in bus_ids]]
    total       = df.sum(axis=1).values
    window_sums = np.convolve(total, np.ones(100, dtype=float), mode="valid")
    best_start  = int(np.argmax(window_sums))
    top100      = df.iloc[best_start : best_start + 100].values.astype(float)
    print(f"  worst 100-slot window starts at row {best_start}  "
          f"(total load range [{top100.sum(axis=1).min():.1f}, "
          f"{top100.sum(axis=1).max():.1f}] MW)")
    return top100


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("Loading network …")
    bus_ids, N_BUS, G_min, G_max, G_inc, F_max = load_network(CASE_PATH)

    print("Loading demand and selecting top-100 time slots …")
    P_load = load_demand(DEMAND_PATH, bus_ids)

    HC_agg = float(G_max.sum() - P_load.sum(axis=1).max())
    print(f"\nAggregate HC = {HC_agg:.2f} MW  (uniform across all buses)")

    print("\nComputing PTDF …")
    H, _, _, _ = compute_ptdf(CASE_PATH)
    print(f"  PTDF shape: {H.shape}  (branches × buses)")

    print("Pre-multiplying H @ G_inc …")
    HA_g    = H @ G_inc        # (n_lines, n_gen)
    Hb_all  = H @ P_load.T    # (n_lines, 100)
    rhs_all = P_load.sum(axis=1)  # (100,)

    # ── Parallel solve ─────────────────────────────────────────────────────────
    print(f"\nSolving {N_BUS} buses × 100 time slots "
          f"({N_BUS * 100:,} LPs) with {N_WORKERS} workers …")

    HC_network = np.full(N_BUS, np.nan)  # nan = not yet computed
    bus_id_to_idx = {int(bid): i for i, bid in enumerate(bus_ids)}

    # ── Resume from checkpoint if one exists ───────────────────────────────────
    if os.path.exists(OUT_PATH):
        ckpt = pd.read_csv(OUT_PATH)
        for _, row in ckpt.iterrows():
            bid = int(row["bus_id"])
            if bid in bus_id_to_idx:
                HC_network[bus_id_to_idx[bid]] = float(row["HC_network_MW"])
        n_loaded = int((~np.isnan(HC_network)).sum())
        print(f"  Resumed from checkpoint: {n_loaded}/{N_BUS} buses already done, "
              f"{N_BUS - n_loaded} remaining.")

    done = int((~np.isnan(HC_network)).sum())

    def _save_checkpoint():
        computed = ~np.isnan(HC_network)
        results = pd.DataFrame({
            "bus_id":          bus_ids[computed],
            "HC_aggregate_MW": HC_agg,
            "HC_network_MW":   HC_network[computed],
        })
        results.to_csv(OUT_PATH, index=False)
        print(f"    [checkpoint] {computed.sum()} buses saved → {OUT_PATH}")

    # ── Retry loop: restarts the pool whenever it crashes ──────────────────────
    while True:
        pending = [k for k in range(N_BUS) if np.isnan(HC_network[k])]
        if not pending:
            break

        print(f"  Submitting {len(pending)} buses to {N_WORKERS} workers …")
        try:
            with ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=_worker_init,
                initargs=(HA_g, H, Hb_all, rhs_all, F_max, G_min, G_max),
            ) as pool:
                futures = {pool.submit(_solve_bus_worker, k): k for k in pending}

                for fut in as_completed(futures):
                    k, hc = fut.result()
                    HC_network[k] = hc
                    done += 1

                    elapsed   = time.time() - t0
                    rate      = done / elapsed
                    remaining = (N_BUS - done) / rate if rate > 0 else 0
                    print(f"  {done:4d}/{N_BUS} buses  "
                          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

                    if done % 100 == 0:
                        _save_checkpoint()

        except BrokenProcessPool:
            n_done = int((~np.isnan(HC_network)).sum())
            print(f"\n  [WARNING] Process pool crashed at {n_done}/{N_BUS} buses. "
                  f"Saving checkpoint and restarting pool …")
            _save_checkpoint()
            # loop continues with the remaining (still-NaN) buses

    # ── Final save ─────────────────────────────────────────────────────────────
    _save_checkpoint()
    print(f"\nDone in {time.time() - t0:.1f}s.  Results → {OUT_PATH}")


if __name__ == "__main__":
    main()
