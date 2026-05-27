import pandas as pd
import torch as tr

from _03_graph_construction import graph_data, apply_physics_constraints, baseGNN

# ------------------------------------------------------------------ #
# CONFIGURATION – change these variables to test different results
# ------------------------------------------------------------------ #
CASE        = "texas"                          # "texas" or "western"
SOURCE      = "csv"                            # "csv" or "model"
CSV_PATH    = r"results\texas_v0.csv"          # used when SOURCE = "csv"
MODEL_PATH  = r"ml_pipeline\outputs\checkpoint.pth"  # used when SOURCE = "model"
VERBOSE     = False   # True: print per-bus violation table for failed constraints
# ------------------------------------------------------------------ #

data, ptdf, rhs_all, F_max_branches, Pf_base = graph_data(CASE)

x0         = data["bus"].x
edge_index = data["bus", "wire", "bus"].edge_index
edge_attr  = data["bus", "wire", "bus"].edge_attr
n_buses    = x0.shape[0]

# ------------------------------------------------------------------ #
# Load predictions
# ------------------------------------------------------------------ #

if SOURCE == "csv":
    df = pd.read_csv(CSV_PATH, index_col="bus_id")
    hc_col = next((c for c in ("HC_network_MW", "hosting_capacity") if c in df.columns), None)
    if hc_col is None:
        raise ValueError(f"No HC column found in {CSV_PATH}. Expected 'HC_network_MW' or 'hosting_capacity'.")

    bus_index_file = r"ml_pipeline\csv\_02_busTexas.csv" if CASE == "texas" else r"ml_pipeline\csv\_12_busWestern.csv"
    bus_df = pd.read_csv(bus_index_file, index_col="bus_i")
    aligned = df[hc_col].reindex(bus_df.index).fillna(0.0)
    pred = tr.tensor(aligned.values, dtype=tr.float32).unsqueeze(1)
    source_label = CSV_PATH

elif SOURCE == "model":
    checkpoint = tr.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model = baseGNN(hidden_channels=64, edge_dim=edge_attr.shape[1])
    model.load_state_dict(state_dict)
    model.eval()
    with tr.no_grad():
        pred = model(x0, edge_index, edge_attr).cpu()
    source_label = MODEL_PATH

else:
    raise ValueError(f"SOURCE must be 'csv' or 'model', got '{SOURCE}'.")

# ------------------------------------------------------------------ #
# Apply physics constraints
# ------------------------------------------------------------------ #

penalties = apply_physics_constraints(pred, x0, edge_index, edge_attr, ptdf, rhs_all, F_max_branches, Pf_base)

# ------------------------------------------------------------------ #
# Per-bus violation counts (derived from the same logic as the penalties)
# ------------------------------------------------------------------ #

dL      = pred.squeeze()
G_max   = x0[:, 21]
G_min   = x0[:, 22]

rhs_t          = tr.tensor(rhs_all, dtype=tr.float32)
headroom_bound = (G_max.sum() - rhs_t.max()).item()
gmin_bound     = (G_min.sum() - rhs_t.min()).item()

H       = tr.tensor(ptdf, dtype=tr.float32)
F_max_t = tr.tensor(F_max_branches, dtype=tr.float32)
# DC base flows + proportional generation correction — mirrors apply_physics_constraints.
# delta_flow[b,k] = (H[b,:] @ g_max_frac - H[b,k]) * dL[k]  (redispatch cancels some flow)
Pd_v      = x0[:, 1]
gen_Pg_v  = x0[:, 10]
Pf_DC     = H @ (gen_Pg_v - Pd_v)                                     # [n_branch]
G_max_v   = x0[:, 21]
H_gen_avg = H @ (G_max_v / G_max_v.sum().clamp(min=1.0))              # [n_branch]
new_flow  = Pf_DC[:, None] + (H_gen_avg[:, None] - H) * dL[None, :]  # [n_branch, n_bus]
flow_violated = (new_flow.abs() > F_max_t[:, None]).any(dim=0)  # [n_bus]

per_bus_violations = {
    "headroom": dL > headroom_bound,
    "gen_min":  dL < gmin_bound,
    "flow":     flow_violated,
    "nonneg":   dL < 0,
}

descriptions = {
    "headroom": "Gen upper bound   (HC ≤ sum(G_max) - rhs, generators can ramp up)",
    "gen_min":  "Gen lower bound   (HC ≥ sum(G_min) - rhs, min-gen ≤ base demand)",
    "flow":     "Thermal limits    (edge capacity ≥ HC per bus)",
    "nonneg":   "Non-negativity    (HC ≥ 0 at every bus)",
}

# ------------------------------------------------------------------ #
# Print results
# ------------------------------------------------------------------ #

print(f"\n{'='*60}")
print(f"  Physics constraint report  |  {CASE.upper()}")
print(f"  Source : {source_label}")
print(f"  Buses  : {n_buses:,}")
print(f"{'='*60}")

all_passed = True
for name, penalty in penalties.items():
    loss_val = penalty.item()
    n_viol   = int(per_bus_violations[name].sum().item())
    pct      = 100.0 * n_viol / n_buses
    passed   = loss_val == 0.0
    status   = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"\n  [{status}]  {descriptions[name]}")
    print(f"          loss (mean ReLU violation) = {loss_val:.6f} MW")
    print(f"          buses violated             = {n_viol:,} / {n_buses:,}  ({pct:.1f}%)")

print(f"\n{'='*60}")
overall = "ALL CONSTRAINTS SATISFIED" if all_passed else "ONE OR MORE CONSTRAINTS VIOLATED"
print(f"  OVERALL: {overall}")
print(f"{'='*60}\n")

# ------------------------------------------------------------------ #
# Verbose: per-bus table for any failing constraint
# ------------------------------------------------------------------ #

if VERBOSE:
    bus_index_file = r"ml_pipeline\csv\_02_busTexas.csv" if CASE == "texas" else r"ml_pipeline\csv\_12_busWestern.csv"
    bus_df = pd.read_csv(bus_index_file, index_col="bus_i")

    report = pd.DataFrame({
        "bus_id":    bus_df.index,
        "pred_HC_MW": dL.detach().numpy(),
    })
    for name, viol_mask in per_bus_violations.items():
        report[f"violates_{name}"] = viol_mask.numpy()

    any_violation = report[[f"violates_{k}" for k in per_bus_violations]].any(axis=1)
    flagged = report[any_violation].reset_index(drop=True)

    if flagged.empty:
        print("  Verbose: no per-bus violations found.\n")
    else:
        print(f"  Verbose: {len(flagged)} buses with at least one violation\n")
        print(flagged.to_string(index=False))
        print()
