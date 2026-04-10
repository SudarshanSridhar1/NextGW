import os
import time
import numpy as np
import scipy.io
import pandas as pd
import cvxpy as cp

# ---------------------------------------------------------------------
# Optional: point to your Gurobi license
# ---------------------------------------------------------------------
os.environ["GRB_LICENSE_FILE"] = "/home/hari/gurobi.lic"

from ptdf import compute_ptdf

# ---------------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------------
CASE_PATH   = "case_western.mat"
DEMAND_PATH = "western_demand.csv"          # can also be .csv.gz if valid
OUT_PATH    = "hosting_capacity_western.csv"

# number of time steps used in the study
N_TIMES = 10

# choose the 100-step window with highest total load, like your earlier code
USE_PEAK_WINDOW = True

# checkpoint frequency
SAVE_EVERY = 1

# optional bus range for testing; set to None for full run
START_BUS = 0
END_BUS   = None   # e.g. 100 for testing, None for all

# solver settings
GUROBI_THREADS = 8
VERBOSE = False


# ---------------------------------------------------------------------
# Load network and fix Fmax = 0 issue
# ---------------------------------------------------------------------
def load_network(case_path):
    mat = scipy.io.loadmat(case_path)
    mpc = mat["mpc"][0, 0]

    bus_arr    = mpc["bus"]
    gen_arr    = mpc["gen"]
    branch_arr = mpc["branch"]

    bus_ids = bus_arr[:, 0].astype(int)
    n_bus   = len(bus_ids)
    bus_idx = {bid: i for i, bid in enumerate(bus_ids)}

    # active generators only
    active_gen = gen_arr[:, 7] == 1
    gen_a = gen_arr[active_gen]

    gen_bus = gen_a[:, 0].astype(int)
    g_max   = gen_a[:, 8].astype(float).copy()
    g_min   = gen_a[:, 9].astype(float).copy()

    gen_idx = np.array([bus_idx[b] for b in gen_bus], dtype=int)

    g_inc = np.zeros((n_bus, len(g_max)), dtype=float)
    for j, bi in enumerate(gen_idx):
        g_inc[bi, j] = 1.0

    # active branches only
    active_br = branch_arr[:, 10] == 1
    branch_a  = branch_arr[active_br].copy()

    # FIX: keep only branches with positive ratings
    f_all = branch_a[:, 5].astype(float)
    keep  = f_all > 0

    branch_f = branch_a[keep].copy()
    f_max    = branch_f[:, 5].astype(float).copy()

    print(f"Total buses: {n_bus}")
    print(f"Active generators: {len(g_max)}")
    print(f"Active lines: {len(branch_a)}")
    print(f"Constrained lines (Fmax > 0): {len(f_max)}")
    print(f"Any zero limits left? {np.any(f_max == 0)}")

    return bus_ids, g_min, g_max, g_inc, keep, f_max


# ---------------------------------------------------------------------
# Load demand
# ---------------------------------------------------------------------
def load_demand(demand_path, bus_ids, n_times=100, use_peak_window=True):
    df = pd.read_csv(demand_path, index_col=0)
    df.columns = df.columns.astype(str)

    # align exactly to bus order in the MATPOWER case
    df = df.reindex(columns=[str(b) for b in bus_ids], fill_value=0.0)

    if use_peak_window:
        total = df.sum(axis=1).values.astype(float)

        if len(total) < n_times:
            raise ValueError(f"demand file has only {len(total)} rows, but N_TIMES={n_times}")

        window_sums = np.convolve(total, np.ones(n_times), mode="valid")
        best_start = int(np.argmax(window_sums))
        df_use = df.iloc[best_start:best_start + n_times]
        print(f"Using peak window: rows {best_start} to {best_start + n_times - 1}")
    else:
        if len(df) < n_times:
            raise ValueError(f"demand file has only {len(df)} rows, but N_TIMES={n_times}")
        df_use = df.iloc[:n_times]
        print(f"Using first {n_times} rows")

    return df_use.values.astype(float)


# ---------------------------------------------------------------------
# Build one reusable optimization model
# ---------------------------------------------------------------------
def build_reusable_model(ha_g, f_max, g_min, g_max):
    """
    Model for one bus k and one time t:
        maximize dL
        s.t. sum(g) == rhs + dL
             g_min <= g <= g_max
             ha_g @ g - hb - hk*dL <= f_max
             ha_g @ g - hb - hk*dL >= -f_max

    hk, hb, rhs are Parameters, so we reuse the same model.
    """
    n_line, n_gen = ha_g.shape

    g  = cp.Variable(n_gen)
    dL = cp.Variable(nonneg=True)

    hk  = cp.Parameter(n_line)   # PTDF column for bus k
    hb  = cp.Parameter(n_line)   # H @ load_t
    rhs = cp.Parameter()         # total load_t

    flow = ha_g @ g - hb - cp.multiply(hk, dL)

    constraints = [
        cp.sum(g) == rhs + dL,
        g >= g_min,
        g <= g_max,
        flow <= f_max,
        flow >= -f_max,
    ]

    problem = cp.Problem(cp.Maximize(dL), constraints)

    return {
        "problem": problem,
        "g": g,
        "dL": dL,
        "hk": hk,
        "hb": hb,
        "rhs": rhs,
    }


# ---------------------------------------------------------------------
# Solve one (bus, time) instance using the reusable model
# ---------------------------------------------------------------------
def solve_once(model_data, hk_val, hb_val, rhs_val):
    model_data["hk"].value  = hk_val
    model_data["hb"].value  = hb_val
    model_data["rhs"].value = float(rhs_val)

    problem = model_data["problem"]

    problem.solve(
        solver=cp.GUROBI,
        verbose=VERBOSE,
        Threads=GUROBI_THREADS,
        OutputFlag=0,
        warm_start=True,
    )

    # occasionally Gurobi reports infeasible_or_unbounded; retry once
    if problem.status == "infeasible_or_unbounded":
        problem.solve(
            solver=cp.GUROBI,
            verbose=VERBOSE,
            Threads=GUROBI_THREADS,
            OutputFlag=0,
            reoptimize=True,
            warm_start=True,
        )

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return np.nan, problem.status

    return float(model_data["dL"].value), problem.status


# ---------------------------------------------------------------------
# Main hosting-capacity loop
# ---------------------------------------------------------------------
def compute_hosting(bus_ids, h, ha_g, hb_all, rhs_all, f_max, g_min, g_max,
                    start_bus=0, end_bus=None, out_path="hosting_capacity_western.csv"):
    n_bus = len(bus_ids)
    n_time = hb_all.shape[1]

    if end_bus is None or end_bus > n_bus:
        end_bus = n_bus

    print(f"Bus range: {start_bus} to {end_bus - 1}")
    print(f"Time steps: {n_time}")

    # build reusable model once
    model_data = build_reusable_model(ha_g, f_max, g_min, g_max)

    results = []
    t_start = time.time()

    for k in range(start_bus, end_bus):
        bus_id = int(bus_ids[k])
        hk_val = h[:, k]

        min_hc = np.inf
        worst_t = None
        bad_status = None

        # Heavier load hours first often tighten the min sooner
        time_order = np.argsort(-rhs_all)

        for t in time_order:
            hc_t, status = solve_once(model_data, hk_val, hb_all[:, t], rhs_all[t])

            if np.isnan(hc_t):
                min_hc = np.nan
                bad_status = status
                worst_t = int(t)
                break

            if hc_t < min_hc:
                min_hc = hc_t
                worst_t = int(t)

            # early exit: cannot do better than essentially zero
            if min_hc <= 1e-8:
                min_hc = 0.0
                break

        results.append((bus_id, min_hc))

        if (k - start_bus + 1) % SAVE_EVERY == 0 or k == end_bus - 1:
            df_out = pd.DataFrame(results, columns=["bus_id", "hosting_capacity"])
            df_out.to_csv(out_path, index=False)

            elapsed = time.time() - t_start
            done = k - start_bus + 1
            rate = elapsed / done
            remain = (end_bus - start_bus - done) * rate / 3600.0

            if bad_status is None:
                print(
                    f"[{done}/{end_bus-start_bus}] "
                    f"bus_id={bus_id}, hc={min_hc:.4f}, worst_t={worst_t}, "
                    f"elapsed={elapsed/60:.1f} min, est_remaining={remain:.1f} hr"
                )
            else:
                print(
                    f"[{done}/{end_bus-start_bus}] "
                    f"bus_id={bus_id}, hc=NaN, status={bad_status}, worst_t={worst_t}, "
                    f"elapsed={elapsed/60:.1f} min, est_remaining={remain:.1f} hr"
                )

    return pd.DataFrame(results, columns=["bus_id", "hosting_capacity"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading network …")
    bus_ids, g_min, g_max, g_inc, keep, f_max = load_network(CASE_PATH)

    print("Loading demand …")
    p_load = load_demand(
        DEMAND_PATH,
        bus_ids,
        n_times=N_TIMES,
        use_peak_window=USE_PEAK_WINDOW
    )

    print("Computing PTDF …")
    h_full, _, _, _ = compute_ptdf(CASE_PATH)

    # apply same line filter to PTDF
    h = h_full[keep, :]

    # precompute repeated quantities
    print("Precomputing matrices …")
    ha_g   = h @ g_inc
    hb_all = h @ p_load.T          # shape: (n_line, n_time)
    rhs_all = p_load.sum(axis=1)   # shape: (n_time,)

    print("Starting hosting-capacity solve …")
    df_res = compute_hosting(
        bus_ids=bus_ids,
        h=h,
        ha_g=ha_g,
        hb_all=hb_all,
        rhs_all=rhs_all,
        f_max=f_max,
        g_min=g_min,
        g_max=g_max,
        start_bus=START_BUS,
        end_bus=END_BUS,
        out_path=OUT_PATH,
    )

    df_res.to_csv(OUT_PATH, index=False)
    print(f"\nSaved final results to {OUT_PATH}")
    print(df_res.head())


if __name__ == "__main__":
    main()