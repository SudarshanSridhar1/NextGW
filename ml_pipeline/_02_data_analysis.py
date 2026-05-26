import pandas as pd

df_nodes = pd.read_csv(r"ml_pipeline\csv\busWestern.csv", index_col="bus_i")
df_edges = pd.read_csv(r"ml_pipeline\csv\branchWestern.csv")
df_generators = pd.read_csv(r"ml_pipeline\csv\genWestern.csv")
df_gen_cost = pd.read_csv(r"ml_pipeline\csv\gencostWestern.csv")

datasets = {
    "bus": df_nodes,
    "branch": df_edges,
    "gen": df_generators,
    "gencost": df_gen_cost,
}

def analyze(name, df):
    print(f"\n{'='*60}")
    print(f"  {name.upper()}  —  {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"{'='*60}")

    print("\n--- Basic stats ---")
    print(df.describe().T.to_string())

    zero_counts = (df == 0).sum()
    zero_pct = (zero_counts / len(df) * 100).round(1)
    zero_summary = pd.DataFrame({"zeros": zero_counts, "zero_%": zero_pct})
    zero_summary = zero_summary[zero_summary["zeros"] > 0]
    if not zero_summary.empty:
        print("\n--- Zero counts (columns with at least one zero) ---")
        print(zero_summary.to_string())

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        print("\n--- Null/NaN counts ---")
        print(null_counts.to_string())
    else:
        print("\nNo nulls found.")

#Basic Stats Summary
#Buses: lam_P, lam_Q, mu_Vmax, mu_Vmin, Gs are all zero across all 2000 rows
#Branches: rateB, rateC, angle, angmin, angmax, mu_Sf, mu_St, mu_angmin, mu_angmax are all zero across all 3206 row
#Generators: Pc1, Pc2, Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_q, mu_Pmax, mu_Pmin, mu_Qmax, mu_Qmin are all zero across all 600 rows
#GenCost is virtually useless and likely can be ommitted 

#In addition, Bs from Buses have 92.5% zeros, so they are virtually useless
#All other variables seem to be able to be used

df_nodes.drop(columns=["lam_P", "lam_Q", "mu_Vmax", "mu_Vmin", "Gs", "Bs"], inplace=True)
df_edges.drop(columns=["rateB", "rateC", "angle", "angmin", "angmax", 
                        "mu_Sf", "mu_St", "mu_angmin", "mu_angmax"], inplace=True)
df_generators.drop(columns=["Pc1", "Pc2", "Qc1min", "Qc1max", 
                            "Qc2min", "Qc2max", "ramp_agc", "ramp_10", "ramp_q", "mu_Pmax", "mu_Pmin", "mu_Qmax", "mu_Qmin"], inplace=True)

for name, df in datasets.items():
    analyze(name, df)

#Usable Features:
#Buses: type, Pd, Qd, area, Vm, Va, baseKV, zone, Vmax, Vmin
#Branches: fbus, tbus | r, x, b, rateA, ratio, status, Pf, Qf, Pt, Qt
#Generators: bus | Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, ramp_30, apf