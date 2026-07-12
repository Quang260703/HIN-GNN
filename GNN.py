"""
gnn.py
Heterogeneous GNN encoders for academic expert retrieval (DBLP/SIGWEB graph).

Node types (example): 'author', 'paper', 'venue', 'term'
Edge types (example):
    ('author', 'writes', 'paper')
    ('paper', 'written_by', 'author')
    ('paper', 'published_in', 'venue')
    ('venue', 'publishes', 'paper')
    ('paper', 'has_term', 'term')
    ('term', 'in_paper', 'paper')

Each encoder takes a torch_geometric.data.HeteroData-style input
(x_dict, edge_index_dict) and returns a dict of node embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, HANConv, RGCNConv, Linear


class HGTEncoder(nn.Module):
    """Heterogeneous Graph Transformer encoder."""

    def __init__(self, metadata, hidden_channels=128, out_channels=64,
                 num_heads=4, num_layers=2):
        super().__init__()
        self.metadata = metadata
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                            heads=num_heads)
            self.convs.append(conv)

        self.out_lin = nn.ModuleDict({
            node_type: Linear(hidden_channels, out_channels)
            for node_type in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: F.relu(self.lin_dict[k](x)) for k, x in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return {k: self.out_lin[k](x) for k, x in x_dict.items()}


class HANEncoder(nn.Module):
    """Heterogeneous Attention Network encoder (requires metapaths)."""

    def __init__(self, metadata, hidden_channels=128, out_channels=64,
                 num_heads=4):
        super().__init__()
        self.han_conv = HANConv(in_channels=-1, out_channels=hidden_channels,
                                 metadata=metadata, heads=num_heads,
                                 dropout=0.2)
        self.out_lin = nn.ModuleDict({
            node_type: Linear(hidden_channels, out_channels)
            for node_type in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict):
        out_dict = self.han_conv(x_dict, edge_index_dict)
        return {
            k: self.out_lin[k](v) for k, v in out_dict.items() if v is not None
        }


class RGCNEncoder(nn.Module):
    """Relational GCN encoder. Expects a single homogeneous edge_index
    with an edge_type tensor (convert HeteroData -> homogeneous first)."""

    def __init__(self, in_channels, hidden_channels=128, out_channels=64,
                 num_relations=6, num_layers=2, num_bases=None):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, num_relations,
                      num_bases=num_bases)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations,
                          num_bases=num_bases)
            )
        self.convs.append(
            RGCNConv(hidden_channels, out_channels, num_relations,
                      num_bases=num_bases)
        )

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
        return x


def build_encoder(name, metadata=None, **kwargs):
    """Factory: name in {'hgt', 'han', 'rgcn'}."""
    name = name.lower()
    if name == "hgt":
        return HGTEncoder(metadata, **kwargs)
    if name == "han":
        return HANEncoder(metadata, **kwargs)
    if name == "rgcn":
        return RGCNEncoder(**kwargs)
    raise ValueError(f"Unknown encoder type: {name}")