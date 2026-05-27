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
sys.path.insert(0, parent_dir)

from data.Texas.ptdf import compute_ptdf

def load_demand(demand_path, bus_ids): #From hosting_capacity_parallel_mod.py
    df = pd.read_csv(demand_path, index_col=0, parse_dates=True)
    df = df[[str(b) for b in bus_ids]]

    total       = df.sum(axis=1).values
    window_sums = np.convolve(total, np.ones(100), mode="valid")
    best_start  = int(np.argmax(window_sums))

    return df.iloc[best_start : best_start + 100].values.astype(float)

def graph_data(case="texas"):
    if case == "western":
        df_edges = pd.read_csv(os.path.join(current_dir, "csv", "_11_branchWestern.csv"))
        df_nodes = pd.read_csv(os.path.join(current_dir, "csv", "_12_busWestern.csv"), index_col="bus_i")
        df_generators = pd.read_csv(os.path.join(current_dir, "csv", "_13_genWestern.csv"))
    else:  # texas
        df_edges = pd.read_csv(os.path.join(current_dir, "csv", "_01_branchTexas.csv"))
        df_nodes = pd.read_csv(os.path.join(current_dir, "csv", "_02_busTexas.csv"), index_col="bus_i")
        df_generators = pd.read_csv(os.path.join(current_dir, "csv", "_03_genTexas.csv"))
    
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
        ptdf, _, _, _ = compute_ptdf(os.path.join(current_dir, "csv", "_00_case_texas.mat")) #TODO: Try Top-K for GNN
        #print(ptdf.shape)
        ptdf_scalar = ptdf.sum(axis=0)
        #print(ptdf_scalar.shape)
    #TODO: Add Western/Eastern cases
    node_data["PTDF_sum"] = ptdf_scalar
    

    #Demand
    if case == "texas":
        bus_ids = df_nodes.index.astype(int).tolist()
        demand = load_demand(os.path.join(parent_dir, "data", "Texas", "texas_demand.csv.gz"), bus_ids=bus_ids)
        rhs_scalar = demand.sum(axis=0)
        #print(rhs_all.shape)
        #print(rhs_scalar.shape)
    node_data["rhs_scalar"] = rhs_scalar

    #------------------------NODE TENSORS------------------------#
    # G_max(21) G_min(22) appended after existing generator features
    x = tr.tensor(node_data.values, dtype=tr.float32)

    #------------------------TARGETS------------------------#
    if case == "western":
        df_targets = pd.read_csv(os.path.join(current_dir, "csv", "_16_hosting_capacity_western.csv"), index_col="bus_id")
        df_targets = df_targets.reindex(df_nodes.index)
        y = tr.tensor(df_targets["hosting_capacity"].values, dtype=tr.float32).unsqueeze(1)
    elif case == "texas":
        df_targets = pd.read_csv(os.path.join(current_dir, "csv", "_06_texas_v0.csv"), index_col="bus_id")
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

    return data, ptdf, demand, F_max_branches, Pf_base

#data, _, _, _, _ = graph_data("texas", cluster=os.name != 'nt')

def apply_physics_constraints(pred, x0, edge_index, edge_attr,
                              ptdf=None, demand=None,
                              F_max_branches=None, Pf_base=None):
    dL    = pred.squeeze()              # [N_BUS] predicted hosting capacity [MW]
    G_max = x0[:, 21]                   # [N_BUS] sum of Pmax over active gens at each bus
    G_min = x0[:, 22]                   # [N_BUS] sum of Pmin over active gens at each bus

    rhs_all = demand.sum(axis=1)        # [n_timestep] total load per timestep (matches LP rhs_all)
    rhs_t   = tr.tensor(rhs_all, dtype=tr.float32, device=dL.device)

    # 1. sum(g) = rhs + dL, g <= G_max  →  dL <= sum(G_max) - rhs   (binds at peak load)
    loss_headroom = F.relu(dL - (G_max.sum() - rhs_t.max())).mean()
    # 2. sum(g) = rhs + dL, g >= G_min  →  dL >= sum(G_min) - rhs   (binds at min load)
    loss_gmin     = F.relu((G_min.sum() - rhs_t.min()) - dL).mean()

    # 3. Flow constraints. LP: flow = HA_g @ g - Hb_param - H_k * dL,  -F_max <= flow <= F_max.
    #    With dispatch frozen at the MATPOWER base case, HA_g @ g - Hb_param collapses to Pf_base.
    #    Per-bus interpretation: dL[k] is what bus k could host alone; broadcast over all buses at once
    #    so column k of flow_per_bus = Pf_base - H[:, k] * dL[k] mirrors the LP's k-th bus problem.
    H         = tr.tensor(ptdf,           dtype=tr.float32, device=dL.device)  # [n_branch, N_BUS]
    F_max_t   = tr.tensor(F_max_branches, dtype=tr.float32, device=dL.device)  # [n_branch]
    Pf_base_t = tr.tensor(Pf_base,        dtype=tr.float32, device=dL.device)  # [n_branch]

    flow_per_bus = Pf_base_t[:, None] - H * dL[None, :]                        # [n_branch, N_BUS]
    loss_flow    = F.relu(flow_per_bus.abs() - F_max_t[:, None]).mean()

    if loss_flow < 0.21: #Magic number of loss for optimal
        loss_flow = 0.21 - (0.21 - loss_flow) * 0.9
    # 4. Non-negativity: dL >= 0  (LP: dL = cp.Variable(nonneg=True))
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
    def __init__(self, hidden_channels, edge_dim, dropout=0.1, dropout_base=0):
        super(baseGNN, self).__init__()
        self.conv1 = gnn.TransformerConv(-1, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv2 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv3 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv4 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv5 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv6 = gnn.TransformerConv(hidden_channels, 1, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout_base)
    def set_constraints(self, ptdf, rhs_all, F_max_branches=None, Pf_base=None):
        self.ptdf = ptdf
        self.rhs_all = rhs_all
        self.F_max_branches = F_max_branches
        self.Pf_base = Pf_base
    def forward(self, x, edge_index, edge_attr):
        x0 = x
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv5(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv6(x, edge_index, edge_attr)
        self.physics_penalty = apply_physics_constraints(
            x, x0, edge_index, edge_attr,
            self.ptdf, self.rhs_all,
            getattr(self, 'F_max_branches', None),
            getattr(self, 'Pf_base', None)
        )
        return x

class virtualNodeGNN(nn.Module):
    def __init__(self, hidden_channels, edge_dim, dropout=0.1, dropout_base=0):
        super(virtualNodeGNN, self).__init__()
        self.conv1 = gnn.TransformerConv(-1, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv2 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv3 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv4 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv5 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.conv6 = gnn.TransformerConv(hidden_channels, 1, edge_dim=edge_dim, heads=4, concat=False, dropout=dropout)
        self.virtualnode = nn.Parameter(tr.zeros(1, hidden_channels))
        self.vn_mlp1 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.vn_mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.vn_mlp3 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.vn_mlp4 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.vn_mlp5 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(dropout_base)
    def set_constraints(self, ptdf, rhs_all, F_max_branches=None, Pf_base=None):
        self.ptdf = ptdf
        self.rhs_all = rhs_all
        self.F_max_branches = F_max_branches
        self.Pf_base = Pf_base
    def _inject_virtual_node(self, x, virtual_state, mlp):
        pooled = x.mean(dim=0, keepdim=True)
        virtual_state = virtual_state + mlp(pooled)
        x = x + virtual_state.expand_as(x)
        return x, virtual_state
    def forward(self, x, edge_index, edge_attr):
        x0 = x
        virtual_state = self.virtualnode

        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=0.01)
        x, virtual_state = self._inject_virtual_node(x, virtual_state, self.vn_mlp1)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=0.01)
        x, virtual_state = self._inject_virtual_node(x, virtual_state, self.vn_mlp2)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=0.01)
        x, virtual_state = self._inject_virtual_node(x, virtual_state, self.vn_mlp3)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=0.01)
        x, virtual_state = self._inject_virtual_node(x, virtual_state, self.vn_mlp4)
        x = self.dropout(x)

        x = self.conv5(x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=0.01)
        x, virtual_state = self._inject_virtual_node(x, virtual_state, self.vn_mlp5)
        x = self.dropout(x)

        x = self.conv6(x, edge_index, edge_attr)

        self.physics_penalty = apply_physics_constraints(
            x, x0, edge_index, edge_attr,
            self.ptdf, self.rhs_all,
            getattr(self, 'F_max_branches', None),
            getattr(self, 'Pf_base', None)
        )
        return x
