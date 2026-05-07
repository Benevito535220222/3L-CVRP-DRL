import gymnasium as gym
import numpy as np
import pandas as pd
import math
from gymnasium import spaces
from numba import jit, njit, prange

@njit(fastmath=True, cache=True)
def _find_lowest_fit_numba(height_map, delivery_order_map, grid_l, grid_w, grid_h, 
                           grid_size, required_support_cells, current_item_rank):
    
    best_stack_pos = (-1, -1, -1)
    min_stack_z = 999999
    
    best_floor_pos = (-1, -1, -1)
    min_floor_y = 999999

    for y in range(0, grid_size - grid_l + 1):
        for x in range(0, grid_size - grid_w + 1):
            
            max_z = 0
            for dy in range(grid_l):
                for dx in range(grid_w):
                    z_val = height_map[y + dy, x + dx]
                    if z_val > max_z:
                        max_z = z_val
            
            if max_z + grid_h > grid_size:
                continue

            lifo_violation = False
            for iy in range(y + grid_l, grid_size): 
                for ix in range(x, x + grid_w):
                    front_rank = delivery_order_map[iy, ix]
                    if front_rank != -1:
                        if front_rank < current_item_rank:
                            lifo_violation = True
                            break
                if lifo_violation: break
            
            if lifo_violation: continue

            if max_z > 0:
                support_count = 0
                vertical_lifo_ok = True
                
                for dy in range(grid_l):
                    for dx in range(grid_w):
                        if height_map[y + dy, x + dx] == max_z:
                            under_rank = delivery_order_map[y + dy, x + dx]
                            if under_rank != -1 and under_rank > current_item_rank:
                                vertical_lifo_ok = False
                                break
                            support_count += 1
                    if not vertical_lifo_ok: break
                
                if vertical_lifo_ok and support_count >= required_support_cells:
                    if max_z < min_stack_z:
                        min_stack_z = max_z
                        best_stack_pos = (y, x, max_z)
                    elif max_z == min_stack_z:
                        if y < best_stack_pos[0]:
                            best_stack_pos = (y, x, max_z)
            
            else:
                if y < min_floor_y:
                    min_floor_y = y
                    best_floor_pos = (y, x, 0)
                elif y == min_floor_y:
                    if x < best_floor_pos[1]: 
                        best_floor_pos = (y, x, 0)

    if best_stack_pos[0] != -1:
        return best_stack_pos
    else:
        return best_floor_pos

@njit(fastmath=True, cache=True)
def _numba_process_obs(
    num_real_items, items_state, anchor, 
    last_id_int, count_in_bin, total_needed, 
    id_indices_flat, id_offsets
):
    forced_id_mask = np.zeros(num_real_items, dtype=np.float32)
    allowed_dest_mask = np.zeros(num_real_items, dtype=np.float32)
    dists = np.full(num_real_items, 999.0, dtype=np.float32)
    min_dist = 999.0

    if last_id_int != -1 and count_in_bin < total_needed:
        start_idx = id_offsets[last_id_int]
        end_idx = id_offsets[last_id_int + 1]
        for i in range(start_idx, end_idx):
            idx = id_indices_flat[i]
            if items_state[idx, 0] > 0.5:
                forced_id_mask[idx] = 1.0

    for i in range(num_real_items):
        if items_state[i, 0] > 0.5:
            dx = items_state[i, 6] - anchor[0]
            dy = items_state[i, 7] - anchor[1]
            d = np.sqrt(dx*dx + dy*dy)
            
            items_state[i, 4] = d / 1.414
            dists[i] = d
            if d < min_dist:
                min_dist = d

    if min_dist < 900:
        tolerance = 0.001
        for i in range(num_real_items):
            if items_state[i, 0] > 0.5 and dists[i] <= (min_dist + tolerance):
                allowed_dest_mask[i] = 1.0
                
    return forced_id_mask, allowed_dest_mask

@njit(fastmath=True, cache=True)
def _calculate_step_metrics(
    item_vol, bin_vol, item_weight, max_weight, 
    z_coord, grid_size, 
    real_l, real_w,
    orig_l, orig_w, orig_h,
    new_dest, prev_dest, depot, rank
):
    util_reward = (item_vol / bin_vol) + (0.5 * item_weight / max_weight)
    
    max_surf = max(orig_l * orig_w, orig_l * orig_h, orig_w * orig_h)
    orientation_bonus = 0.3 * ((real_l * real_w) / max_surf)
    
    stack_bonus = 0.2 * (z_coord / grid_size) if z_coord > 0 else 0.0
    
    if rank == 0:
        dist_added = 2 * np.sqrt(np.sum((new_dest - depot)**2))
    else:
        d_depot_to_new = np.sqrt(np.sum((new_dest - depot)**2))
        d_new_to_prev = np.sqrt(np.sum((new_dest - prev_dest)**2))
        d_depot_to_prev = np.sqrt(np.sum((prev_dest - depot)**2))
        dist_added = d_depot_to_new + d_new_to_prev - d_depot_to_prev
        
    reward = util_reward + orientation_bonus + stack_bonus - (0.15 * dist_added)
    return reward

@njit(parallel=True, fastmath=True)
def _numba_generate_mask(
    num_real_items, num_orientations, action_space_n,
    items_state, grid_dims_lookup, height_map, delivery_map, 
    grid_size, support_threshold, current_rank,
    allowed_destinations_mask,
    forced_id_mask
):
    mask = np.zeros(action_space_n, dtype=np.float32)
    
    is_forcing = np.any(forced_id_mask)

    for idx in prange(num_real_items):
        if items_state[idx, 0] < 0.5:
            continue
        
        if is_forcing and forced_id_mask[idx] == 0:
            continue

        if allowed_destinations_mask[idx] == 0:
            continue

        for orient in range(num_orientations):
            gl, gw, gh = grid_dims_lookup[idx, orient]
            req_sup = int(np.ceil((gl * gw) * support_threshold))
            
            pos = _find_lowest_fit_numba(
                height_map, delivery_map, gl, gw, gh, 
                grid_size, req_sup, current_rank
            )
            
            if pos[0] != -1:
                mask[idx * num_orientations + orient] = 1
                
    return mask

class BinPacking3DEnv(gym.Env):

    def __init__(self, csv_path, bin_dims=(100, 70, 70), grid_size=32, 
                 max_weight=700, support_threshold=0.75, max_items=251, 
                 depot_coords=(0.0, 0.0)):
        super().__init__()
        
        self.bin_dims = np.array(bin_dims, dtype=np.float32)
        self.grid_size = grid_size
        self.max_weight = max_weight
        self.support_threshold = support_threshold
        self.max_items = max_items
        self.bin_volume = np.prod(self.bin_dims)
        self.grid_units = self.bin_dims / self.grid_size
        self.inv_grid_size = 1.0 / self.grid_size
        self.depot_coords = np.array(depot_coords, dtype=np.float32)

        df = pd.read_csv(csv_path)
        df = df.loc[df.index.repeat(df['quantity'])].reset_index(drop=True)
        
        unique_ids = df['id'].unique() 
        self.num_unique_ids = len(unique_ids)
        self.id_to_int_map = {id_str: i for i, id_str in enumerate(unique_ids)}
        self.int_to_id_map = {i: id_str for id_str, i in self.id_to_int_map.items()}

        df['id_int'] = df['id'].map(self.id_to_int_map)
        
        self.id_total_counts = df.groupby('id_int').size().sort_index().values.astype(np.int32)

        self.dest_min = df[['dest_x', 'dest_y']].min().values
        self.dest_max = df[['dest_x', 'dest_y']].max().values
        self.dest_range = self.dest_max - self.dest_min
        self.dest_range[self.dest_range == 0] = 1.0
        
        self.depot_norm = np.array([
            (self.depot_coords[0] - self.dest_min[0]) / self.dest_range[0],
            (self.depot_coords[1] - self.dest_min[1]) / self.dest_range[1]
        ], dtype=np.float32)
        
        df = df.sort_values(by=['dest_x', 'dest_y']).reset_index(drop=True)
        self.raw_items = df.iloc[:max_items].copy()
        self.num_real_items = len(self.raw_items)
        
        self.num_orientations = 6
        self.grid_dims_lookup = np.zeros((self.num_real_items, 6, 3), dtype=np.int32)
        self.real_dims_lookup = np.zeros((self.num_real_items, 6, 3), dtype=np.float32)
        
        for i in range(self.num_real_items):
            l, w, h = self.raw_items.iloc[i][['length', 'width', 'height']]
            rots = [(l,w,h), (l,h,w), (w,l,h), (w,h,l), (h,l,w), (h,w,l)]
            for o, d in enumerate(rots):
                self.real_dims_lookup[i, o] = d
                self.grid_dims_lookup[i, o] = [max(1, math.ceil(d[j]/self.grid_units[j])) for j in range(3)]

        self.action_space = spaces.Discrete(self.max_items * self.num_orientations)
        self.observation_space = spaces.Dict({
            "height_map": spaces.Box(0, 1, (grid_size, grid_size, 1), np.float32),
            "items_state": spaces.Box(0, 1, (max_items, 8), np.float32),
            "action_mask": spaces.Box(0, 1, (self.action_space.n,), np.float32)
        })

        self.global_shipped_indices = set()

        self.raw_items = df.iloc[:max_items].copy()
        self.num_real_items = len(self.raw_items)
        self.id_int_to_indices = self.raw_items.groupby('id_int').indices

        self.id_indices_flat = []
        self.id_offsets = [0]
        all_ids = sorted(self.id_int_to_indices.keys())
        for id_key in all_ids:
            indices = self.id_int_to_indices[id_key]
            self.id_indices_flat.extend(indices)
            self.id_offsets.append(len(self.id_indices_flat))

        self.id_indices_flat = np.array(self.id_indices_flat, dtype=np.int32)
        self.id_offsets = np.array(self.id_offsets, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.delivery_map = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)
        self.current_weight = 0.0
        self.current_volume = 0.0
        self.packed_items = []
        
        self.packed_dests_history = np.zeros((self.max_items, 2), dtype=np.float32)
        
        self.items_state = np.zeros((self.max_items, 8), dtype=np.float32)
        
        raw_values = self.raw_items[['length', 'width', 'height', 'fragile', 'dest_x', 'dest_y']].values
        
        lengths_n = raw_values[:, 0] / self.bin_dims[0]
        widths_n  = raw_values[:, 1] / self.bin_dims[1]
        heights_n = raw_values[:, 2] / self.bin_dims[2]
        dests_x_n = (raw_values[:, 4] - self.dest_min[0]) / self.dest_range[0]
        dests_y_n = (raw_values[:, 5] - self.dest_min[1]) / self.dest_range[1]
        
        for i in range(self.num_real_items):
            if i not in self.global_shipped_indices:
                self.items_state[i, 0] = 1.0
                self.items_state[i, 1] = lengths_n[i]
                self.items_state[i, 2] = widths_n[i]
                self.items_state[i, 3] = heights_n[i]
                self.items_state[i, 5] = raw_values[i, 3]
                self.items_state[i, 6] = dests_x_n[i]
                self.items_state[i, 7] = dests_y_n[i]

        self.current_anchor = self.depot_norm.copy()
        
        return self._get_obs(), {}

    def _get_obs(self):
        if not self.packed_items:
            anchor = self.depot_norm
            last_id_int = -1
            count_in_bin = 0
            total_needed = 0
        else:
            last_p = self.packed_items[-1]
            anchor = np.array([last_p['dest_x_norm'], last_p['dest_y_norm']], dtype=np.float32)
            last_id_int = last_p['id_int']
            count_in_bin = sum(1 for p in self.packed_items if p['id_int'] == last_id_int)
            total_needed = self.id_total_counts[last_id_int]

        forced_id_mask, allowed_dest_mask = _numba_process_obs(
            self.num_real_items, 
            self.items_state, 
            anchor, 
            last_id_int, 
            count_in_bin, 
            total_needed,
            self.id_indices_flat,
            self.id_offsets
        )

        mask = _numba_generate_mask(
            self.num_real_items, 
            self.num_orientations, 
            self.action_space.n,
            self.items_state, 
            self.grid_dims_lookup, 
            self.height_map, 
            self.delivery_map, 
            self.grid_size, 
            self.support_threshold, 
            len(self.packed_items), 
            allowed_dest_mask, 
            forced_id_mask 
        )

        self._cached_mask = mask
        return {
            "height_map": (self.height_map.astype(np.float32) * self.inv_grid_size)[:, :, np.newaxis],
            "items_state": self.items_state.copy(),
            "action_mask": mask
        }
    
    def _to_real_coords(self, grid_pos, grid_dims):
        real_pos = np.array(grid_pos, dtype=np.float32) * self.grid_units
        real_dims = np.array(grid_dims, dtype=np.float32) * self.grid_units
        return real_pos, real_dims

    def step(self, action):
        idx = action // self.num_orientations
        orient = action % self.num_orientations
        
        if idx >= self.num_real_items or self.items_state[idx, 0] < 0.5:
            return self._get_obs(), -2.0, True, False, {"reason": "invalid_action"}

        item_data = self.raw_items.iloc[idx]
        gl, gw, gh = self.grid_dims_lookup[idx, orient]
        rl, rw, rh = self.real_dims_lookup[idx, orient]
        weight = item_data['weight']
        
        if self.current_weight + weight > self.max_weight:
            return self._get_obs(), -1.0, True, False, {"reason": "weight_limit"}

        current_rank = len(self.packed_items)
        pos = _find_lowest_fit_numba(
            self.height_map, self.delivery_map, gl, gw, gh, 
            self.grid_size, int(np.ceil((gl * gw) * self.support_threshold)), current_rank
        )
        
        if pos[0] == -1:
            return self._get_obs(), -1.0, True, False, {"reason": "no_fit"}

        y, x, z = pos
        is_fragile = item_data['fragile'] > 0.5
        self.delivery_map[y:y+gl, x:x+gw] = current_rank
        self.height_map[y:y+gl, x:x+gw] = self.grid_size if is_fragile else z + gh
        
        new_x_norm, new_y_norm = self.items_state[idx, 6], self.items_state[idx, 7]
        prev_dest = self.depot_norm if current_rank == 0 else \
                    np.array([self.packed_items[-1]['dest_x_norm'], self.packed_items[-1]['dest_y_norm']], dtype=np.float32)

        reward = _calculate_step_metrics(
            rl * rw * rh, self.bin_volume, weight, self.max_weight,
            z, self.grid_size, rl, rw,
            item_data['length'], item_data['width'], item_data['height'],
            np.array([new_x_norm, new_y_norm], dtype=np.float32), prev_dest, self.depot_norm, current_rank
        )

        real_pos, real_dims_visual = self._to_real_coords((y, x, z), (gl, gw, gh))

        self.items_state[idx, 0] = 0.0 
        self.current_weight += weight
        self.current_volume += (rl * rw * rh)

        self.packed_items.append({
            'idx': idx, 'id_int': int(item_data['id_int']), 'id': item_data['id'], 'pos': (y, x, z), 'orient': orient, 
            'real_pos': real_pos, 'dims': real_dims_visual,
            'dest_x': item_data['dest_x'], 'dest_y': item_data['dest_y'], 
            'dest_x_norm': new_x_norm, 'dest_y_norm': new_y_norm, 
            'fragile': int(is_fragile), 'delivery_order': current_rank, 'weight': item_data['weight'],
            'length': item_data['length'], 'width': item_data['width'], 'height': item_data['height']
        })
        next_obs = self._get_obs()
        done = (self.current_weight >= self.max_weight) or (np.sum(next_obs["action_mask"]) == 0)
        
        return next_obs, float(reward), done, False, {"pack_ratio": self.current_volume / self.bin_volume}
    
    def _rebuild_maps(self):
        self.height_map.fill(0)
        self.delivery_map.fill(-1)
        self.current_weight = 0.0
        self.current_volume = 0.0

        for i, p in enumerate(self.packed_items):
            p['delivery_order'] = i 
            
            y, x, z = p['pos']
            gl, gw, gh = self.grid_dims_lookup[p['idx'], p['orient']]
            
            self.delivery_map[y:y+gl, x:x+gw] = i
            if p['fragile'] > 0.5:
                self.height_map[y:y+gl, x:x+gw] = self.grid_size
            else:
                self.height_map[y:y+gl, x:x+gw] = z + gh
                
            self.current_weight += p['weight']
            self.current_volume += np.prod(p['dims'])

    def finalize_bin(self):
        if not self.packed_items: 
            return []

        packed_id_counts = np.zeros(self.num_unique_ids, dtype=np.int32)
        for p in self.packed_items:
            packed_id_counts[p['id_int']] += 1

        should_remove_id_mask = (packed_id_counts > 0) & (packed_id_counts < self.id_total_counts)
        
        if np.any(should_remove_id_mask):
            new_packed = []
            removed_indices = []
            
            for p in self.packed_items:
                if should_remove_id_mask[p['id_int']]:
                    self.items_state[p['idx'], 0] = 1.0
                    removed_indices.append(p['idx'])
                else:
                    new_packed.append(p)
            
            self.packed_items = new_packed
            self._rebuild_maps()

        depot = np.array([self.depot_coords[0], self.depot_coords[1]])
        
        if self.packed_items:
            delivery_coords = np.array([[p['dest_x'], p['dest_y']] for p in self.packed_items[::-1]])
            full_path = np.vstack([depot, delivery_coords, depot])
            
            total_dist = 0.0
            for i in range(len(full_path) - 1):
                total_dist += np.linalg.norm(full_path[i] - full_path[i+1])
            self.last_bin_distance = total_dist
            
            for p in self.packed_items:
                self.global_shipped_indices.add(p['idx'])
        else:
            self.last_bin_distance = 0.0
            
        return self.packed_items
    
    def action_masks(self):
        return self._cached_mask
  