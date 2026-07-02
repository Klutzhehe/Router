import random
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class Pin:
    id: int
    component_id: int
    local_x: int
    local_y: int
    global_x: int
    global_y: int
    net_id: int
    layer: int       # Pin copper layer, usually 0 (top) or full/through-hole
    pad_shape: int   # circular=0, rectangular=1, oval=2

@dataclass
class Component:
    id: int
    name: str
    x: int           # bottom-left grid x
    y: int           # bottom-left grid y
    width: int
    height: int
    rotation: int    # 0, 90, 180, 270 degrees
    pins: List[Pin] = field(default_factory=list)

@dataclass
class Net:
    id: int
    name: str
    pin_ids: List[int]
    net_class: str = 'signal'
    is_diff_pair: bool = False
    diff_pair_id: Optional[int] = None
    target_length: float = 0.0          # in mm
    length_tolerance: float = 9999.0     # in mm
    matched_group_id: Optional[int] = None

@dataclass
class Obstacle:
    x: int
    y: int
    width: int
    height: int
    layer: int       # -1 for all layers, or specific layer index

@dataclass
class BoardConfig:
    board_width: int = 500              # 50mm at 0.1mm/px
    board_height: int = 500             # 50mm at 0.1mm/px
    num_nets: int = 5
    num_layers: int = 2
    num_components: int = 4
    obstacle_density: float = 0.1
    num_keep_out_zones: int = 0
    diff_pairs: bool = False
    num_diff_pairs: int = 0
    length_tolerance_mm: float = 1.0
    length_matching: bool = False
    matched_group_size: int = 0
    net_classes: List[str] = field(default_factory=lambda: ['signal'])
    seed: Optional[int] = None

@dataclass
class Board:
    width: int
    height: int
    num_layers: int
    components: List[Component]
    nets: List[Net]
    pins: Dict[int, Pin] # pin_id -> Pin
    obstacles: List[Obstacle]
    keep_out_zones: List[Obstacle]
    design_rules: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Board':
        components = []
        for c in data['components']:
            pins = [Pin(**p) for p in c['pins']]
            c_copy = dict(c)
            c_copy['pins'] = pins
            components.append(Component(**c_copy))
            
        pins = {int(k): Pin(**v) for k, v in data['pins'].items()}
        nets = [Net(**n) for n in data['nets']]
        obstacles = [Obstacle(**o) for o in data['obstacles']]
        keep_out_zones = [Obstacle(**o) for o in data['keep_out_zones']]
        
        return cls(
            width=data['width'],
            height=data['height'],
            num_layers=data['num_layers'],
            components=components,
            nets=nets,
            pins=pins,
            obstacles=obstacles,
            keep_out_zones=keep_out_zones,
            design_rules=data['design_rules']
        )

class BoardGenerator:
    def __init__(self):
        pass

    @classmethod
    def from_curriculum_stage(cls, stage_config: Dict[str, Any]) -> BoardConfig:
        board_gen_cfg = stage_config.get('board_generator', {})
        
        # Resolve ranges to scalar configs
        def resolve_val(val, default):
            if isinstance(val, list):
                return random.randint(val[0], val[1])
            return val if val is not None else default
            
        width, height = 400, 400
        size_range = board_gen_cfg.get('board_size_range')
        if size_range:
            width = resolve_val(size_range, 400)
            height = width # Keep square for simplicity or resolve separately
            
        return BoardConfig(
            board_width=width,
            board_height=height,
            num_nets=resolve_val(board_gen_cfg.get('num_nets_range'), resolve_val(board_gen_cfg.get('num_nets'), 5)),
            num_layers=resolve_val(board_gen_cfg.get('num_layers_range'), board_gen_cfg.get('num_layers', 2)),
            num_components=resolve_val(board_gen_cfg.get('num_components_range'), board_gen_cfg.get('num_components', 4)),
            obstacle_density=board_gen_cfg.get('obstacle_density', 0.1),
            num_keep_out_zones=resolve_val(board_gen_cfg.get('num_keep_out_zones_range'), board_gen_cfg.get('num_keep_out_zones', 0) if board_gen_cfg.get('keep_out_zones') else 0),
            diff_pairs=board_gen_cfg.get('diff_pairs', False),
            num_diff_pairs=resolve_val(board_gen_cfg.get('num_diff_pairs_range'), board_gen_cfg.get('num_diff_pairs', 0)),
            length_matching=board_gen_cfg.get('length_matching', False),
            length_tolerance_mm=board_gen_cfg.get('length_tolerance_mm', 1.0),
            net_classes=board_gen_cfg.get('net_classes', ['signal'])
        )

    def generate(self, config: BoardConfig) -> Board:
        if config.seed is not None:
            random.seed(config.seed)
            
        # Standard design rules
        design_rules = {
            'default': {'width': 0.15, 'clearance': 0.15, 'via_drill': 0.3, 'via_annular': 0.15},
            'power': {'width': 0.3, 'clearance': 0.2, 'via_drill': 0.4, 'via_annular': 0.2},
            'high_speed': {'width': 0.12, 'clearance': 0.12, 'via_drill': 0.25, 'via_annular': 0.12}
        }
        
        # 1. Generate components and place them without overlap
        components = []
        board_margin = 30
        pin_counter = 0
        
        for c_idx in range(config.num_components):
            max_allowed_w = config.board_width - 2 * board_margin
            max_allowed_h = config.board_height - 2 * board_margin
            
            # Find all package options that can fit
            valid_packages = []
            
            # DIP options
            dip_pins_opts = [p for p in [8, 14, 16, 20] if 60 <= max_allowed_w and int((p / 2) * 25.4) <= max_allowed_h]
            if dip_pins_opts:
                valid_packages.append(('DIP', dip_pins_opts))
                
            # QFP options
            qfp_pins_opts = [p for p in [16, 32, 48] if (30 + (p // 4) * 25) <= min(max_allowed_w, max_allowed_h)]
            if qfp_pins_opts:
                valid_packages.append(('QFP', qfp_pins_opts))
                
            # BGA options
            bga_grid_opts = [g for g in [4, 6, 8] if (20 + g * 10) <= min(max_allowed_w, max_allowed_h)]
            if bga_grid_opts:
                valid_packages.append(('BGA', bga_grid_opts))
                
            # SOT option (always fits since it is 30x20)
            if 30 <= max_allowed_w and 20 <= max_allowed_h:
                valid_packages.append(('SOT', [3]))
                
            # Connector options
            conn_pins_opts = [p for p in [4, 6, 8, 10] if 20 <= max_allowed_w and (p * 25) <= max_allowed_h]
            if conn_pins_opts:
                valid_packages.append(('connector', conn_pins_opts))
                
            # Fallback if board is extremely small (SOT-like small component)
            if not valid_packages:
                pkg_type = 'SOT'
                num_pins = 3
                w_cells, h_cells = 30, 20
            else:
                pkg_type, pin_opts = random.choice(valid_packages)
                num_pins = random.choice(pin_opts)
                if pkg_type == 'DIP':
                    w_cells = 60
                    h_cells = int((num_pins / 2) * 25.4)
                elif pkg_type == 'QFP':
                    w_cells = 30 + (num_pins // 4) * 25
                    h_cells = w_cells
                elif pkg_type == 'BGA':
                    grid_side = num_pins
                    num_pins = grid_side ** 2
                    w_cells = 20 + grid_side * 10
                    h_cells = w_cells
                elif pkg_type == 'SOT':
                    w_cells = 30
                    h_cells = 20
                else: # connector
                    w_cells = 20
                    h_cells = num_pins * 25
                
            rotation = random.choice([0, 90, 180, 270])
            
            # Rotated bounding box dimensions
            w_cells_placed = h_cells if rotation in [90, 270] else w_cells
            h_cells_placed = w_cells if rotation in [90, 270] else h_cells
            
            # Place component avoiding overlaps
            max_placement_attempts = 100
            placed = False
            for _ in range(max_placement_attempts):
                max_x = max(board_margin, config.board_width - w_cells_placed - board_margin)
                max_y = max(board_margin, config.board_height - h_cells_placed - board_margin)
                cx = random.randint(board_margin, max_x)
                cy = random.randint(board_margin, max_y)
                
                # Check overlap
                overlap = False
                for ex_c in components:
                    if not (cx + w_cells_placed < ex_c.x or cx > ex_c.x + ex_c.width or
                            cy + h_cells_placed < ex_c.y or cy > ex_c.y + ex_c.height):
                        overlap = True
                        break
                if not overlap:
                    # Place component
                    comp = Component(id=c_idx, name=f"U{c_idx+1}", x=cx, y=cy, width=w_cells_placed, height=h_cells_placed, rotation=rotation)
                    
                    # Generate component pins
                    self._generate_pins(comp, pkg_type, num_pins, pin_counter)
                    pin_counter += len(comp.pins)
                    components.append(comp)
                    placed = True
                    break
                    
        # Re-assign component IDs sequentially to prevent gaps from failed placements
        for new_id, comp in enumerate(components):
            comp.id = new_id
            for pin in comp.pins:
                pin.component_id = new_id

        # Collect all pins
        all_pins = {}
        for comp in components:
            for pin in comp.pins:
                all_pins[pin.id] = pin
                
        # 2. Assign pins to nets
        pin_pool = list(all_pins.keys())
        random.shuffle(pin_pool)
        
        nets = []
        net_idx = 1
        
        # Generate differential pairs if enabled
        diff_pair_counter = 0
        if config.diff_pairs and config.num_diff_pairs > 0:
            for _ in range(config.num_diff_pairs):
                # We need two pairs of adjacent pins on two different components
                if len(components) >= 2:
                    c1, c2 = random.sample(components, 2)
                    if len(c1.pins) >= 2 and len(c2.pins) >= 2:
                        p1_a, p1_b = c1.pins[:2]
                        p2_a, p2_b = c2.pins[:2]
                        
                        # Remove from general pin pool if they are there
                        for p in [p1_a, p1_b, p2_a, p2_b]:
                            if p.id in pin_pool:
                                pin_pool.remove(p.id)
                                
                        # Create Positive Net
                        net_p = Net(
                            id=net_idx, name=f"DIFF_P_{diff_pair_counter}",
                            pin_ids=[p1_a.id, p2_a.id], net_class='high_speed',
                            is_diff_pair=True, diff_pair_id=net_idx+1
                        )
                        p1_a.net_id = net_p.id
                        p2_a.net_id = net_p.id
                        nets.append(net_p)
                        
                        # Create Negative Net
                        net_n = Net(
                            id=net_idx+1, name=f"DIFF_N_{diff_pair_counter}",
                            pin_ids=[p1_b.id, p2_b.id], net_class='high_speed',
                            is_diff_pair=True, diff_pair_id=net_idx
                        )
                        p1_b.net_id = net_n.id
                        p2_b.net_id = net_n.id
                        nets.append(net_n)
                        
                        net_idx += 2
                        diff_pair_counter += 1
                        
        # Generate length matched group if enabled
        if config.length_matching and config.matched_group_size > 0:
            matched_group_pins = []
            # We want to match several nets
            # Draw pins from pool
            # Let's say we have group size of nets, each net needs 2 pins
            num_matched_nets = config.matched_group_size
            if len(pin_pool) >= num_matched_nets * 2:
                group_id = 1
                target_len = random.uniform(20.0, 50.0) # mm
                for _ in range(num_matched_nets):
                    p_a = pin_pool.pop()
                    p_b = pin_pool.pop()
                    net = Net(
                        id=net_idx, name=f"MATCHED_G{group_id}_{net_idx}",
                        pin_ids=[p_a, p_b], net_class='signal',
                        target_length=target_len, length_tolerance=config.length_tolerance_mm,
                        matched_group_id=group_id
                    )
                    all_pins[p_a].net_id = net.id
                    all_pins[p_b].net_id = net.id
                    nets.append(net)
                    net_idx += 1
                    
        # Group remaining pins into standard nets
        while len(pin_pool) >= 2 and len(nets) < config.num_nets:
            # Create a 2-pin net
            p1 = pin_pool.pop()
            p2 = pin_pool.pop()
            
            # Select random net class
            n_class = random.choice(config.net_classes)
            
            net = Net(id=net_idx, name=f"NET_{net_idx}", pin_ids=[p1, p2], net_class=n_class)
            all_pins[p1].net_id = net.id
            all_pins[p2].net_id = net.id
            nets.append(net)
            net_idx += 1
            
        # Cleanup unassigned pins
        for pid in pin_pool:
            all_pins[pid].net_id = 0 # 0 means unassigned / unconnected
            
        # 3. Generate obstacles & keep out zones
        obstacles = []
        keep_out_zones = []
        
        # Calculate obstacle coverage
        total_area = config.board_width * config.board_height
        target_obstacle_area = total_area * config.obstacle_density
        current_obstacle_area = 0
        
        attempts = 0
        while current_obstacle_area < target_obstacle_area and attempts < 100:
            attempts += 1
            ow = random.randint(15, 60)
            oh = random.randint(15, 60)
            ox = random.randint(0, config.board_width - ow)
            oy = random.randint(0, config.board_height - oh)
            
            # Do not overlap with component pins directly (keep some space)
            collision = False
            for p in all_pins.values():
                if ox - 5 <= p.global_x <= ox + ow + 5 and oy - 5 <= p.global_y <= oy + oh + 5:
                    collision = True
                    break
            if not collision:
                layer = random.randint(-1, config.num_layers - 1)
                obstacles.append(Obstacle(x=ox, y=oy, width=ow, height=oh, layer=layer))
                current_obstacle_area += ow * oh
                
        # Generate keep out zones
        for _ in range(config.num_keep_out_zones):
            kow = random.randint(20, 55)
            koh = random.randint(20, 55)
            kox = random.randint(0, config.board_width - kow)
            koy = random.randint(0, config.board_height - koh)
            keep_out_zones.append(Obstacle(x=kox, y=koy, width=kow, height=koh, layer=-1))
            
        return Board(
            width=config.board_width,
            height=config.board_height,
            num_layers=config.num_layers,
            components=components,
            nets=nets,
            pins=all_pins,
            obstacles=obstacles,
            keep_out_zones=keep_out_zones,
            design_rules=design_rules
        )

    def _generate_pins(self, comp: Component, pkg_type: str, num_pins: int, start_id: int):
        pin_spacing = 25 # 2.54mm pitch (in cells)
        
        # Reconstruct original unrotated dimensions for pin generation mapping
        if comp.rotation in [90, 270]:
            w_orig = comp.height
            h_orig = comp.width
        else:
            w_orig = comp.width
            h_orig = comp.height
            
        if pkg_type == 'DIP':
            # Two vertical rows
            pins_per_row = num_pins // 2
            x_left = 10
            x_right = w_orig - 10
            
            for i in range(pins_per_row):
                y = board_margin = 15 + i * pin_spacing
                # Pin ID, local coordinate relative to comp bottom-left, global coordinate, pad_shape
                self._add_pin(comp, start_id + i, x_left, y, 0, 0)
                self._add_pin(comp, start_id + pins_per_row + i, x_right, y, 0, 0)
                
        elif pkg_type == 'QFP':
            # Four rows on each side
            pins_per_side = num_pins // 4
            side_len = w_orig
            
            # Bottom side
            for i in range(pins_per_side):
                self._add_pin(comp, start_id + i, 15 + i * pin_spacing, 10, 0, 1)
            # Right side
            for i in range(pins_per_side):
                self._add_pin(comp, start_id + pins_per_side + i, side_len - 10, 15 + i * pin_spacing, 0, 1)
            # Top side
            for i in range(pins_per_side):
                self._add_pin(comp, start_id + 2*pins_per_side + i, side_len - 15 - i * pin_spacing, side_len - 10, 0, 1)
            # Left side
            for i in range(pins_per_side):
                self._add_pin(comp, start_id + 3*pins_per_side + i, 10, side_len - 15 - i * pin_spacing, 0, 1)
                
        elif pkg_type == 'BGA':
            # Grid layout
            grid_side = int(math.sqrt(num_pins))
            idx = 0
            for r in range(grid_side):
                for c in range(grid_side):
                    self._add_pin(comp, start_id + idx, 15 + c * 10, 15 + r * 10, 0, 0)
                    idx += 1
                    
        elif pkg_type == 'SOT':
            # 3 pins: 2 on one side, 1 on the other
            self._add_pin(comp, start_id, 5, 5, 0, 1)
            self._add_pin(comp, start_id + 1, w_orig - 5, 5, 0, 1)
            self._add_pin(comp, start_id + 2, w_orig // 2, h_orig - 5, 0, 1)
            
        else: # connector
            # Single vertical row
            for i in range(num_pins):
                self._add_pin(comp, start_id + i, w_orig // 2, 12 + i * pin_spacing, 0, 0)

    def _add_pin(self, comp: Component, pin_id: int, local_x: int, local_y: int, layer: int, pad_shape: int):
        # Calculate global coordinates based on component position and rotation
        gx, gy = self._local_to_global(comp, local_x, local_y)
        pin = Pin(
            id=pin_id,
            component_id=comp.id,
            local_x=local_x,
            local_y=local_y,
            global_x=gx,
            global_y=gy,
            net_id=0,
            layer=layer,
            pad_shape=pad_shape
        )
        comp.pins.append(pin)

    def _local_to_global(self, comp: Component, lx: int, ly: int) -> Tuple[int, int]:
        # Reconstruct original unrotated dimensions
        if comp.rotation in [90, 270]:
            w_orig = comp.height
            h_orig = comp.width
        else:
            w_orig = comp.width
            h_orig = comp.height
            
        # Rotation transformation mapping to [0, comp.width] x [0, comp.height]
        if comp.rotation == 90:
            rx = h_orig - ly
            ry = lx
        elif comp.rotation == 180:
            rx = w_orig - lx
            ry = h_orig - ly
        elif comp.rotation == 270:
            rx = ly
            ry = w_orig - lx
        else: # 0
            rx = lx
            ry = ly
            
        return int(round(comp.x + rx)), int(round(comp.y + ry))
