import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Set, Optional
from pcb_router.data.board_generator import Board, Pin, Component, Net, Obstacle

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None

class GraphBuilder:
    def __init__(self):
        pass

    def build_graph(self, board: Board, routed_nets: Set[int] = None) -> Any:
        """
        Convert Board object to PyTorch Geometric HeteroData
        """
        if routed_nets is None:
            routed_nets = set()
            
        data = HeteroData() if HeteroData is not None else dict()
        
        # 1. Build Pad Nodes
        pads = list(board.pins.values())
        N_pads = len(pads)
        pad_features = []
        pad_id_to_index = {}
        
        max_net_id = max([n.id for n in board.nets]) if board.nets else 1
        
        for idx, pin in enumerate(pads):
            pad_id_to_index[pin.id] = idx
            
            # Normalize positions to [0, 1]
            pos_x_norm = pin.global_x / board.width
            pos_y_norm = pin.global_y / board.height
            net_id_norm = pin.net_id / max_net_id if pin.net_id > 0 else 0.0
            
            # Find if this pad belongs to a net with constraints
            target_len_norm = 0.0
            length_tol_norm = 1.0
            is_diff_pair = 0.0
            matched_group = 0.0
            is_source = 0.0
            is_target = 0.0
            
            if pin.net_id > 0:
                # Find matching Net
                net = next((n for n in board.nets if n.id == pin.net_id), None)
                if net:
                    target_len_norm = net.target_length / 100.0 # scale to mm/100
                    length_tol_norm = net.length_tolerance / 10.0
                    is_diff_pair = 1.0 if net.is_diff_pair else 0.0
                    matched_group = float(net.matched_group_id) if net.matched_group_id else 0.0
                    
                    # Simple rule: first pin is source, others are target
                    if net.pin_ids[0] == pin.id:
                        is_source = 1.0
                    else:
                        is_target = 1.0
                        
            # Shape features
            feat = [
                pos_x_norm, pos_y_norm, net_id_norm,
                float(pin.pad_shape), float(pin.layer) / board.num_layers,
                is_source, is_target, target_len_norm, length_tol_norm,
                is_diff_pair, matched_group
            ]
            pad_features.append(feat)
            
        pad_x = torch.tensor(pad_features, dtype=torch.float)
        
        # 2. Build Component Nodes
        N_comps = len(board.components)
        comp_features = []
        for comp in board.components:
            comp_features.append([
                (comp.x + comp.width/2) / board.width,
                (comp.y + comp.height/2) / board.height,
                comp.width / board.width,
                comp.height / board.height,
                len(comp.pins) / 50.0, # scaled pin count
                float(comp.rotation) / 360.0
            ])
        comp_x = torch.tensor(comp_features, dtype=torch.float) if N_comps > 0 else torch.zeros((0, 6), dtype=torch.float)
        
        # 3. Build Region Nodes (8x8 grid cells for coarse spatial encoding)
        grid_size = 8
        region_features = []
        for r in range(grid_size):
            for c in range(grid_size):
                cx = (c + 0.5) / grid_size
                cy = (r + 0.5) / grid_size
                # Compute static congestion estimate based on pin density
                density = 0.0
                for p in pads:
                    px = p.global_x / board.width
                    py = p.global_y / board.height
                    if abs(px - cx) < 0.15 and abs(py - cy) < 0.15:
                        density += 1.0
                region_features.append([cx, cy, density / N_pads if N_pads > 0 else 0.0, 0.0])
        region_x = torch.tensor(region_features, dtype=torch.float)
        
        # 4. Build Via Nodes (Init empty, populated during step updates)
        via_x = torch.zeros((0, 5), dtype=torch.float)
        
        # Store nodes in HeteroData
        if HeteroData is not None:
            data['pad'].x = pad_x
            data['component'].x = comp_x
            data['region'].x = region_x
            data['via'].x = via_x
        else:
            data['pad'] = {'x': pad_x}
            data['component'] = {'x': comp_x}
            data['region'] = {'x': region_x}
            data['via'] = {'x': via_x}
            
        # 5. Build Edges
        # net_connection edges
        net_edge_index = []
        for net in board.nets:
            if net.id not in routed_nets:
                # Add edges between all pins in same net
                for i in range(len(net.pin_ids)):
                    for j in range(i + 1, len(net.pin_ids)):
                        idx_a = pad_id_to_index.get(net.pin_ids[i])
                        idx_b = pad_id_to_index.get(net.pin_ids[j])
                        if idx_a is not None and idx_b is not None:
                            net_edge_index.append([idx_a, idx_b])
                            net_edge_index.append([idx_b, idx_a])
                            
        net_edge_index = torch.tensor(net_edge_index, dtype=torch.long).t() if net_edge_index else torch.zeros((2, 0), dtype=torch.long)
        
        # spatial_proximity edges (K-nearest neighbors)
        k_neighbors = min(8, N_pads - 1)
        spatial_edge_index = []
        if k_neighbors > 0:
            coords = np.array([[p.global_x, p.global_y] for p in pads])
            for i in range(N_pads):
                dists = np.sum((coords - coords[i]) ** 2, axis=1)
                nearest = np.argsort(dists)[1 : k_neighbors + 1] # exclude self
                for neighbor in nearest:
                    spatial_edge_index.append([i, neighbor])
                    
        spatial_edge_index = torch.tensor(spatial_edge_index, dtype=torch.long).t() if spatial_edge_index else torch.zeros((2, 0), dtype=torch.long)
        
        # component_membership edges
        comp_edge_index = []
        rev_comp_edge_index = []
        for comp in board.components:
            for pin in comp.pins:
                pad_idx = pad_id_to_index.get(pin.id)
                if pad_idx is not None:
                    comp_edge_index.append([pad_idx, comp.id])
                    rev_comp_edge_index.append([comp.id, pad_idx])
                    
        comp_edge_index = torch.tensor(comp_edge_index, dtype=torch.long).t() if comp_edge_index else torch.zeros((2, 0), dtype=torch.long)
        rev_comp_edge_index = torch.tensor(rev_comp_edge_index, dtype=torch.long).t() if rev_comp_edge_index else torch.zeros((2, 0), dtype=torch.long)
        
        # Empty via edges to start
        via_edge_index = torch.zeros((2, 0), dtype=torch.long)
        rev_via_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        if HeteroData is not None:
            data['pad', 'net_connection', 'pad'].edge_index = net_edge_index
            data['pad', 'spatial_proximity', 'pad'].edge_index = spatial_edge_index
            data['pad', 'component_membership', 'component'].edge_index = comp_edge_index
            data['component', 'rev_component_membership', 'pad'].edge_index = rev_comp_edge_index
            
            # Via connections
            data['pad', 'via_connection', 'via'].edge_index = via_edge_index
            data['via', 'rev_via_connection', 'pad'].edge_index = rev_via_edge_index
        else:
            data['pad_net_connection_pad'] = net_edge_index
            data['pad_spatial_proximity_pad'] = spatial_edge_index
            data['pad_component_membership_component'] = comp_edge_index
            data['component_rev_component_membership_pad'] = rev_comp_edge_index
            data['pad_via_connection_via'] = via_edge_index
            data['via_rev_via_connection_pad'] = rev_via_edge_index
            
        return data

    def update_graph(self, graph: Any, board: Board, routed_net_id: int, new_traces: List[Any], new_vias: List[Any]) -> Any:
        """
        Dynamically update features in the HeteroData graph object when a net is routed
        """
        # Clone the graph to prevent in-place modifications and sharing mutated states in buffers
        if hasattr(graph, 'clone'):
            graph = graph.clone()
        else:
            import copy
            graph = copy.deepcopy(graph)

        is_mock = not hasattr(graph, 'edge_index_dict')
        
        # 1. Update pad 'is_source' / 'is_target' to zero if net is routed
        # (This helps GNN identify which pins are still active/unrouted)
        pads = list(board.pins.values())
        
        pad_x = graph['pad'].x if not is_mock else graph['pad']['x']
        # Clone pad_x so indexing modifications do not mutate the original tensor in-place
        pad_x = pad_x.clone()
        
        for idx, pin in enumerate(pads):
            if pin.net_id == routed_net_id:
                # Set is_source (idx 5) and is_target (idx 6) to 0.0
                pad_x[idx, 5] = 0.0
                pad_x[idx, 6] = 0.0
                
        if not is_mock:
            graph['pad'].x = pad_x
        else:
            graph['pad']['x'] = pad_x
            
        # 2. Append new via nodes if present
        if new_vias:
            v_feats = []
            for via in new_vias:
                v_feats.append([
                    via.x / board.width,
                    via.y / board.height,
                    via.from_layer / board.num_layers,
                    via.to_layer / board.num_layers,
                    via.net_id / routed_net_id # normalized
                ])
            v_tensor = torch.tensor(v_feats, dtype=torch.float)
            
            via_x = graph['via'].x if not is_mock else graph['via']['x']
            via_x = torch.cat((via_x, v_tensor), dim=0)
            
            if not is_mock:
                graph['via'].x = via_x
            else:
                graph['via']['x'] = via_x
                
            # Note: in a full implementation, we could also reconstruct pad-via connectivity edges here,
            # but for our GNN-encoder v1, updating nodes is sufficient as spatial features of traces
            # are primarily captured by the rasterized board state (ViT channels).
            
        return graph

