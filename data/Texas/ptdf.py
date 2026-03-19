"""
compute_ptdf(case_path) — DC Power Transfer Distribution Factors

PTDF[l, i] = fraction of 1 MW injected at bus i (withdrawn at slack)
             that flows on branch l.

DC lines are omitted: they carry fixed scheduled power, not governed
by network physics, so they do not appear in the DC PTDF.

Method (standard DC approximation):
  1. Build branch susceptances  b = 1 / (x * tap)
  2. Build Bf  (n_branch × n_bus):  Bf[l, f] = b[l],  Bf[l, t] = -b[l]
  3. Build Bbus (n_bus × n_bus):    diagonal = Σ b,  off-diag = -b
  4. Remove slack row/col → B_red  (1999 × 1999 for Texas)
  5. PTDF[:, non_slack] = Bf[:, non_slack] @ inv(B_red)
     PTDF[:, slack]     = 0
"""

import numpy as np
import scipy.io
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized


def compute_ptdf(case_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    case_path : path to a MATPOWER .mat file (mpc struct)

    Returns
    -------
    ptdf     : (n_branch, n_bus) float64  — PTDF matrix
    bus_ids  : (n_bus,)  int              — bus numbers, matching ptdf columns
    f_buses  : (n_branch,) int            — from-bus of each active branch
    t_buses  : (n_branch,) int            — to-bus  of each active branch
    """
    mat = scipy.io.loadmat(case_path)
    mpc = mat["mpc"][0, 0]

    bus    = mpc["bus"]     # (N, 17)  MATPOWER format
    branch = mpc["branch"]  # (B, 21)

    # MATPOWER column indices (0-based)
    BUS_I    = 0   # bus id
    BUS_TYPE = 1   # 1=PQ, 2=PV, 3=slack
    BR_F     = 0   # from bus
    BR_T     = 1   # to bus
    BR_X     = 3   # series reactance
    BR_RATIO = 8   # transformer tap ratio (0 → treat as 1.0)
    BR_STAT  = 10  # status (1=in service)

    # ── Bus index map ──────────────────────────────────────────────────────
    bus_ids = bus[:, BUS_I].astype(int)
    n_bus   = len(bus_ids)
    bus_idx = {bid: i for i, bid in enumerate(bus_ids)}

    slack_i = int(np.where(bus[:, BUS_TYPE] == 3)[0][0])
    non_slack = [i for i in range(n_bus) if i != slack_i]

    # ── Active branches ────────────────────────────────────────────────────
    active = branch[:, BR_STAT] == 1
    br     = branch[active]
    n_br   = len(br)

    f_idx = np.array([bus_idx[int(b)] for b in br[:, BR_F]])
    t_idx = np.array([bus_idx[int(b)] for b in br[:, BR_T]])

    tap = np.where(br[:, BR_RATIO] == 0, 1.0, br[:, BR_RATIO])
    b   = 1.0 / (br[:, BR_X] * tap)   # susceptance per branch

    f_buses = br[:, BR_F].astype(int)
    t_buses = br[:, BR_T].astype(int)

    # ── Build Bf  (n_br × n_bus) ──────────────────────────────────────────
    rows = np.concatenate([np.arange(n_br), np.arange(n_br)])
    cols = np.concatenate([f_idx, t_idx])
    vals = np.concatenate([b, -b])
    Bf   = csr_matrix((vals, (rows, cols)), shape=(n_br, n_bus))

    # ── Build Bbus (n_bus × n_bus) ────────────────────────────────────────
    r = np.concatenate([f_idx, t_idx, f_idx, t_idx])
    c = np.concatenate([f_idx, t_idx, t_idx, f_idx])
    v = np.concatenate([b,     b,    -b,    -b])
    Bbus = csr_matrix((v, (r, c)), shape=(n_bus, n_bus))

    # ── Reduce: remove slack row/col ──────────────────────────────────────
    B_red   = Bbus[non_slack, :][:, non_slack]          # (n-1, n-1) sparse
    Bf_ns   = Bf[:, non_slack].toarray()                 # (n_br, n-1) dense

    # ── Solve  PTDF_ns = Bf_ns @ B_red^{-1} ──────────────────────────────
    # Equivalent to:  B_red.T @ PTDF_ns.T = Bf_ns.T
    # Use sparse LU factorisation (fast for repeated RHS)
    solve  = factorized(B_red.T.tocsc())                 # LU once
    ptdf_ns = solve(Bf_ns.T).T                           # (n_br, n-1)

    # ── Assemble full PTDF (slack column stays 0) ─────────────────────────
    ptdf = np.zeros((n_br, n_bus), dtype=np.float64)
    ptdf[:, non_slack] = ptdf_ns

    return ptdf, bus_ids, f_buses, t_buses


# ── Quick sanity checks ────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from collections import Counter
    BASE = os.path.dirname(__file__)
    case = os.path.join(BASE, "case_texas.mat")

    print("Computing PTDF for Texas system …")
    ptdf, bus_ids, f_buses, t_buses = compute_ptdf(case)

    print(f"\nPTDF shape : {ptdf.shape}  (branches × buses)")
    passed = True

    # 1. Values must lie in [-1, 1]
    print(f"\n[1] PTDF range: [{ptdf.min():.6f}, {ptdf.max():.6f}]  (must be in [-1, 1])")
    ok = ptdf.min() >= -1 - 1e-9 and ptdf.max() <= 1 + 1e-9
    print(f"    {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # 2. Slack column must be all zeros
    slack_col = np.where(np.all(ptdf == 0, axis=0))[0]
    ok = len(slack_col) >= 1
    print(f"\n[2] Slack column all-zero: {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # 3. Solve residual: B_red.T @ PTDF_ns.T ≈ Bf_ns.T
    #    (re-derive from mpc to double-check)
    mat   = scipy.io.loadmat(case)
    mpc   = mat["mpc"][0, 0]
    bus   = mpc["bus"];  branch = mpc["branch"]
    bus_idx = {int(b): i for i, b in enumerate(bus[:, 0])}
    slack_i = int(np.where(bus[:, 1] == 3)[0][0])
    non_slack = [i for i in range(len(bus_ids)) if i != slack_i]
    active = branch[:, 10] == 1;  br = branch[active]
    tap = np.where(br[:, 8] == 0, 1.0, br[:, 8])
    b   = 1.0 / (br[:, 3] * tap)
    f_idx = np.array([bus_idx[int(x)] for x in br[:, 0]])
    t_idx = np.array([bus_idx[int(x)] for x in br[:, 1]])
    n_br, n_bus = len(br), len(bus_ids)
    Bf   = csr_matrix((np.concatenate([b, -b]),
                       (np.concatenate([np.arange(n_br)]*2),
                        np.concatenate([f_idx, t_idx]))), shape=(n_br, n_bus))
    r = np.concatenate([f_idx, t_idx, f_idx, t_idx])
    c = np.concatenate([f_idx, t_idx, t_idx, f_idx])
    v = np.concatenate([b, b, -b, -b])
    Bbus = csr_matrix((v, (r, c)), shape=(n_bus, n_bus))
    B_red  = Bbus[non_slack, :][:, non_slack]
    Bf_ns  = Bf[:, non_slack].toarray()
    ptdf_ns = ptdf[:, non_slack]
    residual = np.abs(B_red.T @ ptdf_ns.T - Bf_ns.T).max()
    ok = residual < 1e-6
    print(f"\n[3] Solve residual max|B_red.T @ PTDF_ns.T - Bf_ns.T|: {residual:.2e}  (expect < 1e-6)")
    print(f"    {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # 4. Truly radial bus: degree-1 bus should have |PTDF| ≈ 1 on its branch
    #    Degree = total incident branches (from + to)
    degree = Counter(f_idx.tolist() + t_idx.tolist())
    radial_bus_idx = [i for i, d in degree.items() if d == 1]
    if radial_bus_idx:
        bi = radial_bus_idx[0]
        # find the branch touching this bus
        br_idx = np.where((f_idx == bi) | (t_idx == bi))[0][0]
        val = ptdf[br_idx, bi]
        ok  = abs(abs(val) - 1.0) < 1e-6
        print(f"\n[4] Radial bus (idx {bi}, bus {bus_ids[bi]}): "
              f"|PTDF| on its branch = {abs(val):.6f}  (expect 1.0)")
        print(f"    {'PASS' if ok else 'FAIL'}")
        passed &= ok
    else:
        print("\n[4] No degree-1 (radial) bus found — skipping check")

    print(f"\n{'All checks passed.' if passed else 'Some checks FAILED.'}")
