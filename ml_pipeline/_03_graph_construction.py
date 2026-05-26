import pandas as pd
import torch as tr
import torch_geometric
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.transforms as T
import torch.nn.functional as F
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.Texas.ptdf import compute_ptdf

def load_demand(demand_path, bus_ids): #From hosting_capacity_parallel_mod.py
    df = pd.read_csv(demand_path, index_col=0, parse_dates=True)
    df = df[[str(b) for b in bus_ids]]

    total       = df.sum(axis=1).values
    window_sums = np.convolve(total, np.ones(100), mode="valid")
    best_start  = int(np.argmax(window_sums))

    return df.iloc[best_start : best_start + 100].values.astype(float)

def graph_data(case="texas", cluster=True):
    if case == "western" and not cluster:
        df_edges = pd.read_csv(rf"ml_pipeline\csv\_11_branchWestern.csv")
        df_nodes = pd.read_csv(rf"ml_pipeline\csv\_12_busWestern.csv", index_col="bus_i")
        df_generators = pd.read_csv(rf"ml_pipeline\csv\_13_genWestern.csv")
    elif case == "texas" and not cluster:
        df_edges = pd.read_csv(rf"ml_pipeline\csv\_01_branchTexas.csv")
        df_nodes = pd.read_csv(rf"ml_pipeline\csv\_02_busTexas.csv", index_col="bus_i")
        df_generators = pd.read_csv(rf"ml_pipeline\csv\_03_genTexas.csv")
    elif case == "western" and not cluster:
        df_edges = pd.read_csv(rf"csv/_11_branchWestern.csv")
        df_nodes = pd.read_csv(rf"csv/_12_busWestern.csv", index_col="bus_i")
        df_generators = pd.read_csv(rf"csv/_13_genWestern.csv")
    else:
        df_edges = pd.read_csv(rf"csv/_01_branchTexas.csv")
        df_nodes = pd.read_csv(rf"csv/_02_busTexas.csv", index_col="bus_i")
        df_generators = pd.read_csv(rf"csv/_03_genTexas.csv")
    
    df_nodes.drop(columns=["lam_P", "lam_Q", "mu_Vmax", "mu_Vmin", "Gs", "Bs"], inplace=True)
    df_edges.drop(columns=["rateB", "rateC", "angle", "angmin", "angmax", 
                            "mu_Sf", "mu_St", "mu_angmin", "mu_angmax"], inplace=True)
    df_generators.drop(columns=["Pc1", "Pc2", "Qc1min", "Qc1max", 
                                "Qc2min", "Qc2max", "ramp_agc", "ramp_10", "ramp_q", 
                                "mu_Pmax", "mu_Pmin", "mu_Qmax", "mu_Qmin"], inplace=True)
    
    bus_features = ["type", "Pd", "Qd", "area", "Vm", "Va", "baseKV", "zone", "Vmax", "Vmin"]
    edge_features = ["r", "x", "b", "rateA", "ratio", "status", "Pf", "Qf", "Pt", "Qt"]
    generator_features = ["Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin", "ramp_30", "apf"]

    connections_node = ["bus_i"]
    generator_node = ["bus"]
    connections_edge = ["fbus", "tbus"]

    generator_by_bus = (
		df_generators.groupby("bus")[generator_features]
		.sum()
		.rename(columns={col: f"gen_{col}" for col in generator_features})
	)

    node_data = df_nodes[bus_features]
    node_data = node_data.join(generator_by_bus, how="left")
    for col in generator_by_bus.columns.tolist():
        node_data[col] = node_data[col].fillna(0.0)

    # G_max, G_min: sum Pmax/Pmin over active generators (status==1) per bus.
    # Format note: per-bus aggregation vs per-generator in the LP.
    active_gens = df_generators[df_generators["status"] == 1]
    gen_capacity = (
        active_gens.groupby("bus")[["Pmax", "Pmin"]]
        .sum()
        .reindex(df_nodes.index, fill_value=0.0)
    )
    node_data["G_max"] = gen_capacity["Pmax"].values  # [N_BUS]
    node_data["G_min"] = gen_capacity["Pmin"].values  # [N_BUS]

    #PTDF
    if case == "texas":
        ptdf, _, _, _ = compute_ptdf(r"ml_pipeline\csv\_00_case_texas.mat") #TODO: Try Top-K for GNN
        #print(ptdf.shape)
        ptdf_scalar = ptdf.sum(axis=0)
        #print(ptdf_scalar.shape)
    #TODO: Add Western/Eastern cases
    node_data["PTDF_sum"] = ptdf_scalar
    

    #Demand
    if case == "texas":
        bus_ids = df_nodes.index.astype(int).tolist()
        demand = load_demand(r"data\Texas\texas_demand.csv.gz", bus_ids=bus_ids)
        rhs_scalar = demand.sum(axis=0)
        rhs_all = demand.sum(axis=1)
        #print(rhs_all.shape)
        #print(rhs_scalar.shape)
    node_data["rhs_scalar"] = rhs_scalar

    #------------------------NODE TENSORS------------------------#
    # G_max(21) G_min(22) appended after existing generator features
    x = tr.tensor(node_data.values, dtype=tr.float32)

    #------------------------TARGETS------------------------#
    if case == "western":
        if cluster:
            df_targets = pd.read_csv(rf"csv/_16_hosting_capacity_western.csv", index_col="bus_id")
        else:
            df_targets = pd.read_csv(rf"ml_pipeline\csv\_16_hosting_capacity_western.csv", index_col="bus_id")
        df_targets = df_targets.reindex(df_nodes.index)
        y = tr.tensor(df_targets["hosting_capacity"].values, dtype=tr.float32).unsqueeze(1)
    elif case == "texas":
        if cluster:
            df_targets = pd.read_csv(rf"csv/_06_texas_v0.csv", index_col="bus_id")
        else:
            df_targets = pd.read_csv(rf"ml_pipeline\csv\_06_texas_v0.csv", index_col="bus_id")
        df_targets = df_targets.reindex(df_nodes.index)
        y = tr.tensor(df_targets["HC_network_MW"].values, dtype=tr.float32).unsqueeze(1)

    #------------------------EDGES------------------------#
    bus_to_index = {bus_id: idx for idx, bus_id in enumerate(df_nodes.index)}
    edge_index = df_edges[connections_edge].replace(bus_to_index).values.T
    edge_index = tr.tensor(edge_index, dtype=tr.long)
    edge_attr = tr.tensor(df_edges[edge_features].values, dtype=tr.float32)

    edge_margin = (edge_attr[:, 3] - edge_attr[:, 6].abs()).clamp(min=0.0)  # rateA - |Pf|, available margin on each edge
    edge_attr = tr.cat([edge_attr, edge_margin.unsqueeze(1)], dim=1)  # append margin as additional edge feature

    #------------------------HETERODATA------------------------#
    data = torch_geometric.data.HeteroData()
    data["bus"].x = x
    data["bus"].y = y
    data["bus", "wire", "bus"].edge_index = edge_index
    data["bus", "wire", "bus"].edge_attr = edge_attr
    data = T.ToUndirected()(data)
    # F_max (thermal rating) is edge_attr[:, 3] (rateA); G_max/G_min are x[:, 21]/x[:, 22]; edge_margin is edge_attr[:, 10]

    # Active branch thermal limits and base flows in PTDF row ordering.
    # compute_ptdf filters branch status==1 in MATPOWER case order; df_edges preserves the same order.
    active_br_mask = df_edges["status"] == 1
    F_max_branches = df_edges.loc[active_br_mask, "rateA"].values.astype(float)  # [n_active_branch]
    # DC base flows: H @ (gen_Pg - Pd), consistent with the LP's DC model.
    # avoids AC/DC mismatch from the MATPOWER Pf column.
    g_per_bus = active_gens.groupby("bus")["Pg"].sum().reindex(df_nodes.index, fill_value=0.0).values
    Pd_arr    = df_nodes["Pd"].values
    Pf_base   = ptdf @ (g_per_bus - Pd_arr)               # [n_active_branch] DC base flows

    return data, ptdf, rhs_all, F_max_branches, Pf_base

data, _, _, _, _ = graph_data("texas", cluster=False)

def apply_physics_constraints(pred, x0, edge_index, edge_attr,
                              ptdf=None, rhs_all=None,
                              F_max_branches=None, Pf_base=None):
    dL    = pred.squeeze()   # [N_BUS] predicted hosting capacity [MW]
    G_max = x0[:, 21]        # max generation per bus [MW]  (active gens, summed)
    G_min = x0[:, 22]        # min generation per bus [MW]  (active gens, summed)

    if rhs_all is not None:
        rhs_t = tr.tensor(rhs_all, dtype=tr.float32, device=dL.device)  # [n_timestep]
        # 1. sum(g) = rhs + dL, g <= G_max  →  dL <= sum(G_max) - rhs  (binding at peak load)
        loss_headroom = F.relu(dL - (G_max.sum() - rhs_t.max())).mean()
        # 2. sum(g) = rhs + dL, g >= G_min  →  dL >= sum(G_min) - rhs  (binding at min load)
        loss_gmin = F.relu((G_min.sum() - rhs_t.min()) - dL).mean()
    else:
        gen_Pg = x0[:, 10]
        system_headroom = (G_max - gen_Pg).clamp(min=0.0).sum()
        loss_headroom = F.relu(dL - system_headroom).mean()
        rhs_min_gmin = (gen_Pg - G_min).clamp(min=0.0).sum()
        loss_gmin = F.relu(-rhs_min_gmin - dL).mean()

    # 3. Thermal limits via full PTDF: flow[b,k] = Pf_base[b] - H[b,k]*dL[k]  (LP sign convention:
    #    dL is additional load, subtracts from net injection).  Falls back to local edge-margin proxy
    #    when PTDF data is unavailable.
    if ptdf is not None and F_max_branches is not None:
        H       = tr.tensor(ptdf, dtype=tr.float32, device=dL.device)             # [n_branch, n_bus]
        F_max_t = tr.tensor(F_max_branches, dtype=tr.float32, device=dL.device)  # [n_branch]
        # DC base flows: H @ (gen_Pg - Pd).  Matches the LP's H @ (G_inc @ g - P_load)
        # formulation exactly, avoiding AC/DC mismatch from the MATPOWER Pf column.
        Pd_v    = x0[:, 1].to(H.device)                  # [n_bus] base demand
        gen_Pg  = x0[:, 10].to(H.device)                 # [n_bus] base generation
        Pf_DC   = H @ (gen_Pg - Pd_v)                    # [n_branch]
        # Proportional generation correction: adding dL[k] requires sum(delta_g)=dL[k].
        # Distributing delta_g proportional to G_max gives the net flow change per bus:
        #   delta_flow[b,k] = (H[b,:] @ g_max_frac - H[b,k]) * dL[k]
        G_max_n   = x0[:, 21].to(H.device)
        H_gen_avg = H @ (G_max_n / G_max_n.sum().clamp(min=1.0))                  # [n_branch]
        new_flow  = Pf_DC[:, None] + (H_gen_avg[:, None] - H) * dL[None, :]      # [n_branch, n_bus]
        loss_flow = F.relu(new_flow.abs() - F_max_t[:, None]).mean()
    else:
        edge_margin = edge_attr[:, 10]
        avail = tr.zeros(dL.shape[0], device=dL.device)
        avail.scatter_add_(0, edge_index[0], edge_margin)
        loss_flow = F.relu(dL - avail).mean()

    # 4. Non-negativity: dL >= 0
    loss_nonneg = F.relu(-dL).mean()

    return {
        "headroom": loss_headroom,
        "gen_min":  loss_gmin,
        "flow":     loss_flow,
        "nonneg":   loss_nonneg,
    }

#n_gen = len(G_max)
#H_k   = H[:, k]

#g  = cp.Variable(n_gen)
#dL = cp.Variable(nonneg=True)

#Hb_param  = cp.Parameter(len(F_max))
#rhs_param = cp.Parameter()

#flow = HA_g @ g - Hb_param - H_k * dL

#constraints = [
#    cp.sum(g) == rhs_param + dL,
#    g >= G_min,
#    g <= G_max,
#    flow <= F_max,
#    flow >= -F_max,
#]


class baseGNN(nn.Module): 
    def __init__(self, hidden_channels, edge_dim):
        super(baseGNN, self).__init__()
        self.conv1 = gnn.TransformerConv(-1, hidden_channels, edge_dim=edge_dim)
        self.conv2 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv3 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv4 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv5 = gnn.TransformerConv(hidden_channels, 1, edge_dim=edge_dim)
    def set_constraints(self, ptdf, rhs_all, F_max_branches=None, Pf_base=None):
        self.ptdf = ptdf
        self.rhs_all = rhs_all
        self.F_max_branches = F_max_branches
        self.Pf_base = Pf_base
    def forward(self, x, edge_index, edge_attr):
        x0 = x
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_attr)
        self.physics_penalty = apply_physics_constraints(
            x, x0, edge_index, edge_attr,
            self.ptdf, self.rhs_all,
            getattr(self, 'F_max_branches', None),
            getattr(self, 'Pf_base', None)
        )
        return x
