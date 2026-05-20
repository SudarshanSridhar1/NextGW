import networkx as nx
import pandas as pd
import torch as tr
import torch_geometric
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.transforms as T

from _03_graph_construction import graph_data

data, y = graph_data("western")


def chain_lengths(data: torch_geometric.data.HeteroData):
    """Return (longest_chain, average_chain) as shortest-path lengths over all node pairs."""
    edge_index = data["bus", "wire", "bus"].edge_index.numpy()
    num_nodes = data["bus"].x.shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))

    # Use the largest connected component if the graph is disconnected
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    longest = nx.diameter(G)
    average = nx.average_shortest_path_length(G)
    return longest, average


def build_graph(data: torch_geometric.data.HeteroData) -> nx.Graph:
    edge_index = data["bus", "wire", "bus"].edge_index.numpy()
    num_nodes = data["bus"].x.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    return G


def plot_buses(data: torch_geometric.data.HeteroData, y: tr.Tensor = None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    G = build_graph(data)

    # Spectral layout initialises spring layout well for large graphs
    pos = nx.spring_layout(G, k=2.5, seed=42, pos=nx.spectral_layout(G))

    node_colors = "steelblue"
    cbar = None
    if y is not None:
        values = y.squeeze().numpy()
        node_colors = values
        vmin, vmax = values.min(), values.max()

    fig, ax = plt.subplots(figsize=(14, 10))

    if y is not None:
        sc = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=8,
            node_color=node_colors, cmap=cm.plasma, vmin=vmin, vmax=vmax
        )
        cbar = plt.colorbar(sc, ax=ax, label="Hosting Capacity")
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=8, node_color=node_colors)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=5, edge_color="purple")

    ax.set_title(f"Bus Network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(r"ml_pipeline\csv\bus_network_western.png", dpi=300)


longest, average = chain_lengths(data)
print(f"Longest chain: {longest}")
print(f"Average chain length: {average:.4f}")

#----TEXAS----#
#Longest Chain: 30
#Average Chain Length: 12.9828
#Nodes: 2000, Edges: 2667

#----WESTERN----#
#Longest Chain: 52
#Average Chain Length: 22.5295
#Nodes: 10024, Edges: 12241

#plot_buses(data, y)