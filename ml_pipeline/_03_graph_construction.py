import pandas as pd
import torch as tr
import torch_geometric
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.transforms as T

def graph_data(case="texas"):
    if case == "western":
        df_edges = pd.read_csv(r"ml_pipeline\csv\_11_branchWestern.csv")
        df_nodes = pd.read_csv(r"ml_pipeline\csv\_12_busWestern.csv", index_col="bus_i")
        df_generators = pd.read_csv(r"ml_pipeline\csv\_13_genWestern.csv")
    elif case == "texas":
        df_edges = pd.read_csv(r"ml_pipeline\csv\_01_branchTexas.csv")
        df_nodes = pd.read_csv(r"ml_pipeline\csv\_02_busTexas.csv", index_col="bus_i")
        df_generators = pd.read_csv(r"ml_pipeline\csv\_03_genTexas.csv")
    
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
    connections_edge = ["fbus, tbus"]

    generator_by_bus = (
		df_generators.groupby("bus")[generator_features]
		.sum()
		.rename(columns={col: f"gen_{col}" for col in generator_features})
	)

    node_data = df_nodes[bus_features]
    node_data = node_data.join(generator_by_bus, how="left")
    for col in generator_by_bus.columns.tolist():
        node_data[col] = node_data[col].fillna(0.0)

    #------------------------NODE TENSORS------------------------#
    x = tr.tensor(node_data.values, dtype=tr.float32)

    #------------------------TARGETS------------------------# #TODO: Fix targets to actual
    df_hc = pd.read_csv(r"ml_pipeline\csv\hosting_capacity_sum_results.csv", index_col="time_slot")
    bus_hc_cols = [col for col in df_hc.columns if col.startswith("dL_bus_")]
    hc_by_bus = df_hc[bus_hc_cols].mean(axis=0)
    hc_by_bus.index = hc_by_bus.index.str.replace("dL_bus_", "").astype(float)
    y_values = hc_by_bus.reindex(df_nodes.index).fillna(0.0).values
    y = tr.tensor(y_values.reshape(-1, 1), dtype=tr.float32)

    #------------------------EDGES------------------------#
    bus_to_index = {bus_id: idx for idx, bus_id in enumerate(df_nodes.index)}
    edge_index = df_edges[["fbus", "tbus"]].replace(bus_to_index).values.T
    edge_index = tr.tensor(edge_index, dtype=tr.long)
    edge_attr = tr.tensor(df_edges[edge_features].values, dtype=tr.float32)

    #------------------------HETERODATA------------------------#
    data = torch_geometric.data.HeteroData()
    data["bus"].x = x
    data["bus", "wire", "bus"].edge_index = edge_index
    data["bus", "wire", "bus"].edge_attr = edge_attr
    data = T.ToUndirected()(data)

    return data, y

data, y = graph_data(case = "texas")

class baseGNN(nn.Module):
	def __init__(self, hidden_channels, edge_dim):
		super(baseGNN, self).__init__()
		self.conv1 = gnn.TransformerConv(-1, hidden_channels, edge_dim=edge_dim)
		self.conv2 = gnn.TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
		self.conv3 = gnn.TransformerConv(hidden_channels, 1, edge_dim=edge_dim) #output vm_pu?
	def forward(self, x, edge_index, edge_attr):
		x = self.conv1(x, edge_index, edge_attr)
		x = x.relu()
		x = self.conv2(x, edge_index, edge_attr)
		x = x.relu()
		x = self.conv3(x, edge_index, edge_attr)
		return x
	
def apply_physics_contraints(x):
    # Placeholder for physics-based constraints
    # For example, you could enforce that voltage magnitudes are within certain limits
    return x

edge_dim = data["bus", "wire", "bus"].edge_attr.shape[1]
base_model = baseGNN(hidden_channels=16, edge_dim=edge_dim)

output = base_model(data["bus"].x, 
					data["bus", "wire", "bus"].edge_index, 
					data["bus", "wire", "bus"].edge_attr)

print(output.shape) #[2000, 1]
