import scipy.io
import pandas as pd
import os

mat = scipy.io.loadmat(r'ml_pipeline\csv\case_western.mat')
mpc = mat['mpc'][0, 0]

# --- bus (2000 x 17): 13 standard + 4 OPF result columns ---
bus_cols = [
    'bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area', 'Vm', 'Va',
    'baseKV', 'zone', 'Vmax', 'Vmin',
    'lam_P', 'lam_Q', 'mu_Vmax', 'mu_Vmin'
]
bus = pd.DataFrame(mpc['bus'], columns=bus_cols)

# --- gen (600 x 25): 21 standard + 4 OPF result columns ---
gen_cols = [
    'bus', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status', 'Pmax', 'Pmin',
    'Pc1', 'Pc2', 'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max',
    'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q', 'apf',
    'mu_Pmax', 'mu_Pmin', 'mu_Qmax', 'mu_Qmin'
]
gen = pd.DataFrame(mpc['gen'], columns=gen_cols)

# --- branch (3206 x 21): 13 standard + 8 OPF result columns ---
branch_cols = [
    'fbus', 'tbus', 'r', 'x', 'b', 'rateA', 'rateB', 'rateC',
    'ratio', 'angle', 'status', 'angmin', 'angmax',
    'Pf', 'Qf', 'Pt', 'Qt', 'mu_Sf', 'mu_St', 'mu_angmin', 'mu_angmax'
]
branch = pd.DataFrame(mpc['branch'], columns=branch_cols)

# --- gencost (600 x 7): polynomial cost model (n=3 → c2, c1, c0) ---
gencost_cols = ['model', 'startup', 'shutdown', 'n', 'c2', 'c1', 'c0']
gencost = pd.DataFrame(mpc['gencost'], columns=gencost_cols)

out_dir = r'ml_pipeline\csv'
os.makedirs(out_dir, exist_ok=True)

bus.to_csv(os.path.join(out_dir, 'busWestern.csv'), index=False)
gen.to_csv(os.path.join(out_dir, 'genWestern.csv'), index=False)
branch.to_csv(os.path.join(out_dir, 'branchWestern.csv'), index=False)
gencost.to_csv(os.path.join(out_dir, 'gencostWestern.csv'), index=False)

print(f"bus:     {bus.shape}")
print(f"gen:     {gen.shape}")
print(f"branch:  {branch.shape}")
print(f"gencost: {gencost.shape}")
print(f"Saved CSVs to {out_dir}/")