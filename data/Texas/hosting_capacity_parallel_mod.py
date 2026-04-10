import os
import sys
import time
import numpy as np
import scipy.io
import pandas as pd
import cvxpy as cp

# 🔥 Ensure Gurobi license is visible
os.environ["GRB_LICENSE_FILE"] = "/home/hari/gurobi.lic"

# ── Paths ─────────────────────────────────────────
TEXAS_DIR   = os.path.dirname(os.path.abspath(__file__))
CASE_PATH   = os.path.join(TEXAS_DIR, "case_texas.mat")
DEMAND_PATH = os.path.join(TEXAS_DIR, "texas_demand.csv.gz")
OUT_PATH    = os.path.join(TEXAS_DIR, "hosting_capacity_results.csv")

sys.path.insert(0, TEXAS_DIR)
from ptdf import compute_ptdf


# ── Solver per bus ─────────────────────────────────
def _solve_bus(k, HA_g, H, Hb_all, rhs_all, F_max, G_min, G_max):
    n_gen = len(G_max)
    H_k   = H[:, k]

    g  = cp.Variable(n_gen)
    dL = cp.Variable(nonneg=True)

    Hb_param  = cp.Parameter(len(F_max))
    rhs_param = cp.Parameter()

    flow = HA_g @ g - Hb_param - H_k * dL

    constraints = [
        cp.sum(g) == rhs_param + dL,
        g >= G_min,
        g <= G_max,
        flow <= F_max,
        flow >= -F_max,
    ]

    problem = cp.Problem(cp.Maximize(dL), constraints)

    min_hc = np.inf

    for t in range(Hb_all.shape[1]):
        Hb_param.value  = Hb_all[:, t]
        rhs_param.value = float(rhs_all[t])

        problem.solve(
            solver=cp.GUROBI,
            verbose=False,
            Threads=8,     # 🔥 parallelism inside Gurobi
            OutputFlag=0   # 🔇 silence logs
        )

        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            min_hc = min(min_hc, float(dL.value))
        else:
            return k, np.nan

    return k, (0.0 if min_hc == np.inf else min_hc)


# ── Data loaders ───────────────────────────────────
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

    GEN_BUS = gen_a[:, 0].astype(int)
    G_max   = gen_a[:, 8].copy()
    G_min   = gen_a[:, 9].copy()

    GEN_IDX = np.array([bus_idx[b] for b in GEN_BUS])

    G_inc = np.zeros((N_BUS, len(G_max)))
    for j, bi in enumerate(GEN_IDX):
        G_inc[bi, j] = 1.0

    active_br = branch_arr[:, 10] == 1
    F_max     = branch_arr[active_br, 5].copy()

    return bus_ids, N_BUS, G_min, G_max, G_inc, F_max


def load_demand(demand_path, bus_ids):
    df = pd.read_csv(demand_path, index_col=0, parse_dates=True)
    df = df[[str(b) for b in bus_ids]]

    total       = df.sum(axis=1).values
    window_sums = np.convolve(total, np.ones(100), mode="valid")
    best_start  = int(np.argmax(window_sums))

    return df.iloc[best_start : best_start + 100].values.astype(float)


# ── Main ───────────────────────────────────────────
def main():
    t0 = time.time()

    print("Loading network …")
    bus_ids, N_BUS, G_min, G_max, G_inc, F_max = load_network(CASE_PATH)

    print("Loading demand …")
    P_load = load_demand(DEMAND_PATH, bus_ids)

    print("Computing PTDF …")
    H, _, _, _ = compute_ptdf(CASE_PATH)

    HA_g    = H @ G_inc
    Hb_all  = H @ P_load.T
    rhs_all = P_load.sum(axis=1)

    results = []

    for k in range(N_BUS):
        print(f"Solving bus {k+1}/{N_BUS}")

        _, hc = _solve_bus(k, HA_g, H, Hb_all, rhs_all, F_max, G_min, G_max)
        results.append((k, hc))

        # 🔥 Save checkpoint after each bus
        df_out = pd.DataFrame(results, columns=["bus_index", "hosting_capacity"])
        df_out["bus_id"] = df_out["bus_index"].map(lambda i: bus_ids[i])
        df_out = df_out[["bus_index", "bus_id", "hosting_capacity"]]

        df_out.to_csv(OUT_PATH, index=False)

        if (k + 1) % 10 == 0:
            print(f"Checkpoint saved at bus {k+1}")

    print(f"\nSaved results to {OUT_PATH}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()