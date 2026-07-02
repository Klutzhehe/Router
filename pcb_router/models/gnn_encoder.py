import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_geometric.nn as geom_nn
    from torch_geometric.data import HeteroData
except ImportError:
    # Fallback/mock logic for local testing without pyg installed
    geom_nn = None
    HeteroData = None

class HeteroGATEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        out_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        pad_in_dim: int = 11,
        via_in_dim: int = 5,
        comp_in_dim: int = 6,
        region_in_dim: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Node input projections
        self.proj = nn.ModuleDict({
            'pad': nn.Linear(pad_in_dim, hidden_dim),
            'via': nn.Linear(via_in_dim, hidden_dim),
            'component': nn.Linear(comp_in_dim, hidden_dim),
            'region': nn.Linear(region_in_dim, hidden_dim)
        })
        
        # Heterogeneous GAT Convolutions
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # Define edge types to perform message passing on
            edge_convs = {}
            # pad <-> pad (net connection)
            edge_convs[('pad', 'net_connection', 'pad')] = geom_nn.GATConv(
                hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False
            ) if geom_nn else None
            # pad <-> pad (spatial proximity)
            edge_convs[('pad', 'spatial_proximity', 'pad')] = geom_nn.GATConv(
                hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False
            ) if geom_nn else None
            # pad -> component
            edge_convs[('pad', 'component_membership', 'component')] = geom_nn.GATConv(
                hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False
            ) if geom_nn else None
            # component -> pad
            edge_convs[('component', 'rev_component_membership', 'pad')] = geom_nn.GATConv(
                hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False
            ) if geom_nn else None
            # pad <-> via (optional, let's include if present)
            edge_convs[('pad', 'via_connection', 'via')] = geom_nn.GATConv(
                hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False
            ) if geom_nn else None
            edge_convs[('via', 'rev_via_connection', 'pad')] = geom_nn.GATConv(
                hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False
            ) if geom_nn else None
            
            # Filter None if geom_nn is not available
            edge_convs = {k: v for k, v in edge_convs.items() if v is not None}
            
            if geom_nn:
                self.convs.append(geom_nn.HeteroConv(edge_convs, aggr='sum'))
        
        # Layer Norms for each node type
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in ['pad', 'via', 'component', 'region']
            })
            for _ in range(num_layers)
        ])
        
        # Output projections to out_dim (384)
        self.out_proj = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, out_dim)
            for node_type in ['pad', 'via', 'component', 'region']
        })
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # 1. Input projection
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.proj:
                h_dict[node_type] = self.proj[node_type](x)
            else:
                h_dict[node_type] = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
                
        # 2. Heterogeneous message passing
        for i in range(self.num_layers):
            if len(self.convs) > i:
                # Store residuals
                res_dict = {k: v for k, v in h_dict.items()}
                
                # Perform convolution
                out_dict = self.convs[i](h_dict, edge_index_dict)
                
                # Apply activation, normalization, dropout, and residual connection
                for node_type in h_dict.keys():
                    if node_type in out_dict:
                        h = out_dict[node_type]
                        h = F.elu(h)
                        h = self.norms[i][node_type](h)
                        h = self.dropout(h)
                        h_dict[node_type] = res_dict[node_type] + h
                        
        # 3. Output projection
        out_embeddings = {}
        for node_type, h in h_dict.items():
            out_embeddings[node_type] = self.out_proj[node_type](h)
            
        return out_embeddings
