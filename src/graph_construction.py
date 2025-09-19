# -*- coding: utf-8 -*-
"""
图构建与增广模块 (Graph Construction & Augmentation)

本模块包含构建和处理图结构的核心逻辑。
主要功能包括：
- EDGE_TYPE_* 常量: 定义了项目中使用的五种边类型。
- GraphConstructor:
  - 核心类，负责将预处理后的数据窗口转换为PyG图对象。
  - 实现静态边（语义相似性、内部连通性）的构建逻辑。
  - 实现物理嵌入的动态边（阴影、植被活动、局部风场）的构建逻辑。
- GraphAugmentor:
  - 可选的图增广类。
  - 能够为图添加额外的结构特征，如节点度、聚类系数、拉普拉斯位置编码等。
"""

import numpy as np
import torch
import math
import networkx as nx
import scipy.sparse.linalg as spla

from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_scatter import scatter_mean

# ==================== 边类型常量定义 ===================
EDGE_TYPE_TREE_ACTIVITY = 0.0
EDGE_TYPE_SHADOW = 1.0
EDGE_TYPE_LOCAL_WIND = 2.0
EDGE_TYPE_SIMILARITY = 3.0
EDGE_TYPE_INTERNAL_N8 = 4.0

# ==================== 图结构构建模块 ===================
class GraphConstructor:
    """根据物理原理和节点特征构建动态异构图。"""
    def __init__(self, **kwargs):
        """
        初始化图构建器。
        所有物理参数和超参数都通过kwargs传入，方便配置。
        """
        self.grid_size = kwargs.get('grid_size', 4)
        self.k_similarity = kwargs.get('k_similarity', 8)
        self.target_attr_index = kwargs.get('target_attr_index', 5)
        self.global_tree_max = kwargs.get('global_tree_max_val', 1.0)
        self.global_building_max = kwargs.get('global_building_max_val', 1.0)
        self.base_building_shadow_radius_max_grids = kwargs.get('base_building_shadow_radius_max_grids', 15)
        self.base_tree_shadow_radius_max_grids = kwargs.get('base_tree_shadow_radius_max_grids', 5)
        self.base_tree_activity_radius_max_grids = kwargs.get('base_tree_activity_radius_max_grids', 4)
        self.wind_effect_on_radius_factor = kwargs.get('wind_effect_on_radius_factor', 0.3)
        self.max_expected_wind_speed = kwargs.get('max_expected_wind_speed', 8.0)
        self.shadow_angular_width_rad = math.radians(kwargs.get('shadow_angular_width_deg', 30.0))
        self.base_edge_weight = kwargs.get('base_edge_weight', 1.0)
        self.distance_decay_factor_per_grid = kwargs.get('distance_decay_factor_per_grid', 0.1)
        self.similarity_dist_decay_factor_per_grid = kwargs.get('similarity_dist_decay_factor_per_grid', 0.005)
        self.actual_shadow_boost_factor = kwargs.get('actual_shadow_boost_factor', 1.05)
        self.tree_activity_height_influence_factor = kwargs.get('tree_activity_height_influence_factor', 0.2)
        self.knn_epsilon = kwargs.get('knn_node_feature_normalization_epsilon', 1e-6)

    def build_graph(self, input_window_base_features, output_window_all_hours,
                      hourly_weather_params: dict, solar_params: dict,
                      target_hour_index_in_day: int) -> Data:
        """
        为单个时间步（小时）构建一个完整的图。

        Args:
            input_window_base_features (np.ndarray): 当前空间窗口的节点基础特征。
            output_window_all_hours (np.ndarray): 当前空间窗口的所有小时的目标变量。
            hourly_weather_params (dict): 当前小时的天气参数。
            solar_params (dict): 当前小时的太阳位置参数。
            target_hour_index_in_day (int): 用于提取目标 `y` 的小时索引。

        Returns:
            torch_geometric.data.Data: 构建好的图数据对象。
        """
        node_features, positions = self._extract_local_node_features(input_window_base_features)
        num_nodes = len(node_features)

        graph_global_env_f = np.array([
            hourly_weather_params['temperature_c'], hourly_weather_params['humidity_percent'],
            hourly_weather_params['wind_speed_mps'], hourly_weather_params['pressure_pa'],
            hourly_weather_params['global_horizontal_radiation_whm2'],
            solar_params['azimuth_rad'], solar_params['elevation_rad']
        ], dtype=np.float32)

        h, w = input_window_base_features.shape[:2]
        hcoords = {(r, c): r * w + c for r in range(h) for c in range(w)}

        ei_dyn, ea_dyn, ew_dyn = self._build_dynamic_local_edges(
            node_features, positions, hcoords, h, w, hourly_weather_params, solar_params
        )
        ei_sim, ea_sim, ew_sim = self._build_edges_topk_feature_similarity(
            node_features, positions, hourly_weather_params
        )
        edge_index, edge_attr, edge_weights = self._merge_undirected_edges(
            ei_dyn, ea_dyn, ew_dyn, ei_sim, ea_sim, ew_sim
        )

        node_targets = self._process_targets(output_window_all_hours, target_hour_index_in_day)
        
        building_mask = (input_window_base_features[:, :, 7].flatten() > 0.5)

        if edge_weights.size > 0:
            edge_attr = np.concatenate([edge_attr, edge_weights.reshape(-1, 1)], axis=-1)
        else:
            edge_attr = np.empty((0, edge_attr.shape[1] + 1), dtype=np.float32)

        return Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index).t().contiguous() if edge_index.size > 0 else torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.FloatTensor(edge_attr),
            edge_weight=torch.FloatTensor(edge_weights),
            y=torch.FloatTensor(node_targets),
            pos=torch.FloatTensor(positions),
            building_mask=torch.BoolTensor(building_mask),
            graph_global_env_features=torch.FloatTensor(graph_global_env_f)
        )

    def _extract_local_node_features(self, window_base_features):
        h, w, c = window_base_features.shape
        features = window_base_features.reshape(-1, c)
        positions = np.array([[c_idx * self.grid_size, r_idx * self.grid_size] for r_idx in range(h) for c_idx in range(w)])
        return features, positions

    def _process_targets(self, output_window_all_hours, target_hour_index):
        target_slice = output_window_all_hours[self.target_attr_index, :, :, target_hour_index]
        return target_slice.reshape(-1, 1)

    def _calculate_edge_weight(self, dist_m, edge_type, is_shadow_interaction=False, tree_height_norm_factor=None):
        weight = self.base_edge_weight
        dist_grids = dist_m / self.grid_size

        if edge_type == EDGE_TYPE_SIMILARITY:
            if self.similarity_dist_decay_factor_per_grid > 0 and dist_m > 0:
                weight /= (1.0 + self.similarity_dist_decay_factor_per_grid * dist_grids)
        else:
            if self.distance_decay_factor_per_grid > 0 and dist_m > 0:
                weight /= (1.0 + self.distance_decay_factor_per_grid * dist_grids)

        if edge_type == EDGE_TYPE_TREE_ACTIVITY and tree_height_norm_factor is not None and self.tree_activity_height_influence_factor > 0:
            modulation = 1.0 + self.tree_activity_height_influence_factor * tree_height_norm_factor
            weight *= modulation
        
        if edge_type == EDGE_TYPE_SHADOW and is_shadow_interaction:
            weight *= self.actual_shadow_boost_factor

        return max(weight, 0.01)

    # ... (Rest of the GraphConstructor methods: _is_internal_node, _build_dynamic_local_edges, 
    #      _build_edges_topk_feature_similarity, _merge_undirected_edges)
    # NOTE: These methods are complex and long. For brevity in this thought block, I'll paste them directly
    # into the final code block. The logic remains unchanged.
    def _is_internal_node(self, r, c, window_h, window_w, hcoords, node_features_local, is_checking_tree: bool):
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr_offset, dc_offset in neighbor_offsets:
            nr, nc = r + dr_offset, c + dc_offset
            if not (0 <= nr < window_h and 0 <= nc < window_w): return False
            neighbor_node_idx = hcoords.get((nr, nc))
            if neighbor_node_idx is None: return False
            
            neighbor_feat = node_features_local[neighbor_node_idx]
            if is_checking_tree:
                if not (neighbor_feat[0] > 0): return False
            else: # is_checking_building
                if not (neighbor_feat[7] > 0.5): return False
        return True

    def _build_dynamic_local_edges(self, node_features_local, positions, hcoords, window_h, window_w, hourly_weather_params: dict, solar_params: dict):
        # This method is very long. I will copy it verbatim from the user's code.
        # It's the core of the physics-based edge creation.
        edge_index_list, edge_attr_list, edge_weights_list = [], [], []
        num_nodes = len(node_features_local)

        sol_elev_rad = solar_params['elevation_rad']
        wind_speed = hourly_weather_params['wind_speed_mps']
        wind_blowing_to_meteo_rad = hourly_weather_params['wind_blowing_to_meteo_rad']
        radiation = hourly_weather_params['global_horizontal_radiation_whm2']
        shadow_main_direction_meteo_rad = (solar_params['azimuth_rad'] + math.pi) % (2 * math.pi)

        for i in range(num_nodes):
            src_feat_local = node_features_local[i]
            src_pos = positions[i]
            src_grid_r, src_grid_c = int(round(src_pos[1]/self.grid_size)), int(round(src_pos[0]/self.grid_size))

            is_tree_node = src_feat_local[0] > 0
            is_building_node = src_feat_local[7] > 0.5
            is_object_node = is_tree_node or is_building_node
            is_internal = False

            if is_object_node:
                is_internal = self._is_internal_node(src_grid_r, src_grid_c, window_h, window_w, hcoords, node_features_local, is_checking_tree=is_tree_node)

            if is_internal:
                for dr_n8 in range(-1, 2):
                    for dc_n8 in range(-1, 2):
                        if dr_n8 == 0 and dc_n8 == 0: continue
                        tgt_grid_r, tgt_grid_c = src_grid_r + dr_n8, src_grid_c + dc_n8
                        if (tgt_grid_r, tgt_grid_c) in hcoords:
                            j = hcoords[(tgt_grid_r, tgt_grid_c)]
                            dist = np.linalg.norm(positions[j] - src_pos)
                            edge_index_list.append([i, j])
                            edge_attr_list.append([dist, float(dc_n8), float(dr_n8), 0.0, EDGE_TYPE_INTERNAL_N8])
                            edge_weights_list.append(self._calculate_edge_weight(dist, EDGE_TYPE_INTERNAL_N8))
            else:
                # Tree Activity Edges
                if is_tree_node:
                    activity_rad_factor = np.clip(radiation / 1000, 0.5, 1.2)
                    radius_grids = max(1, int(round(self.base_tree_activity_radius_max_grids * activity_rad_factor)))
                    normalized_tree_height = src_feat_local[0]
                    for j in range(num_nodes):
                        if i == j: continue
                        dist = np.linalg.norm(positions[j] - src_pos)
                        if dist <= radius_grids * self.grid_size:
                            vec_ij = positions[j] - src_pos
                            dx, dy = vec_ij[0]/self.grid_size, vec_ij[1]/self.grid_size
                            edge_index_list.append([i, j])
                            edge_attr_list.append([dist, dx, dy, 0.0, EDGE_TYPE_TREE_ACTIVITY])
                            edge_weights_list.append(self._calculate_edge_weight(dist, EDGE_TYPE_TREE_ACTIVITY, tree_height_norm_factor=normalized_tree_height))
                
                # Shadow Edges
                if is_object_node and sol_elev_rad > math.radians(1.0):
                    obj_height_norm = src_feat_local[0] if is_tree_node else src_feat_local[6]
                    obj_height_abs = obj_height_norm * (self.global_tree_max if is_tree_node else self.global_building_max)
                    shadow_len_m = obj_height_abs / math.tan(sol_elev_rad) if math.tan(sol_elev_rad) > 1e-6 else 0
                    base_max_grids = self.base_tree_shadow_radius_max_grids if is_tree_node else self.base_building_shadow_radius_max_grids
                    shadow_len_grids = min(int(round(shadow_len_m / self.grid_size)), base_max_grids)

                    for j in range(num_nodes):
                        if i == j: continue
                        vec_ij = positions[j] - src_pos
                        dist = np.linalg.norm(vec_ij)
                        if 0 < dist <= shadow_len_grids * self.grid_size:
                            angle_to_target_meteo_rad = math.atan2(vec_ij[0], vec_ij[1])
                            angle_diff = abs(angle_to_target_meteo_rad - shadow_main_direction_meteo_rad)
                            angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                            if angle_diff <= self.shadow_angular_width_rad / 2.0:
                                dx, dy = vec_ij[0]/self.grid_size, vec_ij[1]/self.grid_size
                                edge_index_list.append([i, j])
                                edge_attr_list.append([dist, dx, dy, 0.0, EDGE_TYPE_SHADOW])
                                edge_weights_list.append(self._calculate_edge_weight(dist, EDGE_TYPE_SHADOW, is_shadow_interaction=True))
                
                # Local Wind Edges
                base_local_radius_grids = 1 if not is_object_node else 2
                for j in range(num_nodes):
                    if i == j: continue
                    vec_ij = positions[j] - src_pos
                    dist = np.linalg.norm(vec_ij)
                    if dist <= base_local_radius_grids * self.grid_size * 1.5: # A bit larger to account for wind effect
                        angle_ij_cartesian_rad = math.atan2(vec_ij[1], vec_ij[0])
                        wind_blowing_to_cartesian_rad = (math.pi/2 - wind_blowing_to_meteo_rad) % (2*math.pi)
                        wind_align_cos = math.cos(angle_ij_cartesian_rad - wind_blowing_to_cartesian_rad)
                        wind_strength_factor = np.clip(wind_speed / self.max_expected_wind_speed, 0, 1)
                        radius_mod = 1.0 + self.wind_effect_on_radius_factor * wind_align_cos * wind_strength_factor
                        effective_dist = dist / radius_mod if radius_mod > 1e-6 else dist * 1e6
                        if effective_dist <= base_local_radius_grids * self.grid_size:
                            dx, dy = vec_ij[0]/self.grid_size, vec_ij[1]/self.grid_size
                            edge_index_list.append([i, j])
                            edge_attr_list.append([dist, dx, dy, wind_align_cos, EDGE_TYPE_LOCAL_WIND])
                            edge_weights_list.append(self._calculate_edge_weight(dist, EDGE_TYPE_LOCAL_WIND))

        if not edge_index_list:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 5), dtype=np.float32), np.empty(0, dtype=np.float32)
        return np.array(edge_index_list), np.array(edge_attr_list), np.array(edge_weights_list)

    def _build_edges_topk_feature_similarity(self, node_features_local, positions, hourly_weather_params: dict):
        num_nodes = len(node_features_local)
        if num_nodes <= 1 or self.k_similarity <= 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 5), dtype=np.float32), np.empty(0, dtype=np.float32)

        mean = np.mean(node_features_local, axis=0, keepdims=True)
        std = np.std(node_features_local, axis=0, keepdims=True)
        norm_features = (node_features_local - mean) / (std + self.knn_epsilon)
        
        dist_mat = pairwise_distances(norm_features, metric='euclidean')
        np.fill_diagonal(dist_mat, np.inf)

        edge_index_list, edge_attr_list, edge_weights_list = [], [], []
        
        for i in range(num_nodes):
            actual_k = min(self.k_similarity, num_nodes - 1)
            if actual_k <= 0: continue
            topk_idx = np.argsort(dist_mat[i])[:actual_k]

            for nbr_idx in topk_idx:
                src_pos, dst_pos = positions[i], positions[nbr_idx]
                dist = np.linalg.norm(dst_pos - src_pos)
                vec_ij = dst_pos - src_pos
                dx_grid = vec_ij[0] / self.grid_size
                dy_grid = vec_ij[1] / self.grid_size
                
                edge_index_list.append([i, nbr_idx])
                edge_attr_list.append([dist, dx_grid, dy_grid, 0.0, EDGE_TYPE_SIMILARITY])
                edge_weights_list.append(self._calculate_edge_weight(dist, EDGE_TYPE_SIMILARITY))

        if not edge_index_list:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 5), dtype=np.float32), np.empty(0, dtype=np.float32)
        return np.array(edge_index_list), np.array(edge_attr_list), np.array(edge_weights_list)

    def _merge_undirected_edges(self, ei1, ea1, ew1, ei2, ea2, ew2):
        if ei1.size == 0:
            edge_index, edge_attr, edge_weights = ei2, ea2, ew2
        elif ei2.size == 0:
            edge_index, edge_attr, edge_weights = ei1, ea1, ew1
        else:
            edge_index = np.concatenate([ei1, ei2], axis=0)
            edge_attr = np.concatenate([ea1, ea2], axis=0)
            edge_weights = np.concatenate([ew1, ew2], axis=0)
        
        if edge_index.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 5), dtype=np.float32), np.empty(0, dtype=np.float32)

        unique_edges = {}
        for idx in range(edge_index.shape[0]):
            src, dst = edge_index[idx, 0], edge_index[idx, 1]
            if src != dst:
                unique_edges[(src, dst)] = (edge_attr[idx], edge_weights[idx])
        
        final_ei, final_ea, final_ew = [], [], []
        processed = set()
        for (s, d), (attr_sd, weight_sd) in unique_edges.items():
            u, v = min(s, d), max(s, d)
            if (u, v) in processed: continue

            final_ei.append([s, d])
            final_ea.append(attr_sd)
            final_ew.append(weight_sd)

            if (d, s) in unique_edges:
                attr_ds, weight_ds = unique_edges[(d, s)]
                final_ei.append([d, s])
                final_ea.append(attr_ds)
                final_ew.append(weight_ds)
            else: # Create reverse edge
                attr_ds = attr_sd.copy()
                attr_ds[1:4] *= -1 # Reverse dx, dy, cos_align
                final_ei.append([d, s])
                final_ea.append(attr_ds)
                final_ew.append(weight_sd) # Keep same weight
            processed.add((u, v))
        
        if not final_ei:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 5), dtype=np.float32), np.empty(0, dtype=np.float32)
        return np.array(final_ei), np.array(final_ea), np.array(final_ew)


# ==================== 图增广模块 ===================
class GraphAugmentor:
    """为图数据对象添加额外的节点或边特征。"""
    def __init__(self, add_neighbor_agg: bool = False, add_edge_diff: bool = False, use_laplacian_pe: bool = False, lap_pe_dim: int = 4):
        self.add_neighbor_agg = add_neighbor_agg
        self.add_edge_diff = add_edge_diff
        self.use_laplacian_pe = use_laplacian_pe
        self.lap_pe_dim = lap_pe_dim
    
    def augment_static(self, data: Data) -> Data:
        if self.add_neighbor_agg: self._add_neighbor_mean_features(data)
        if self.add_edge_diff: self._add_edge_diff(data)
        if self.use_laplacian_pe: self._add_laplacian_positional_encoding(data)
        return data

    def _add_neighbor_mean_features(self, data: Data):
        if data.num_nodes == 0 or data.num_edges == 0: return
        row, col = data.edge_index
        x_mean = scatter_mean(data.x[col].float(), row, dim=0, dim_size=data.num_nodes)
        data.x = torch.cat([data.x, x_mean], dim=1)

    def _add_edge_diff(self, data: Data):
        if data.num_nodes == 0 or data.num_edges == 0 or not hasattr(data, 'edge_attr') or data.edge_attr is None: return
        src_x, dst_x = data.x[data.edge_index[0]], data.x[data.edge_index[1]]
        data.edge_attr = torch.cat([data.edge_attr, src_x - dst_x], dim=1)

    def _add_laplacian_positional_encoding(self, data: Data):
        if data.num_nodes < 3 or data.num_edges == 0 or self.lap_pe_dim <= 0: return
        try:
            G = to_networkx(data, to_undirected=True)
            L = nx.laplacian_matrix(G).astype(float)
            k = min(self.lap_pe_dim, data.num_nodes - 2)
            if k <= 0: return
            
            vals, vecs = spla.eigsh(L, k=k, which='SM', tol=1e-3)
            vecs = vecs[:, np.argsort(vals)]
            lap_pe = torch.from_numpy(vecs).float().to(data.x.device)
            data.x = torch.cat([data.x, lap_pe], dim=1)
        except Exception as e:
            print(f"    警告: 计算拉普拉斯位置编码失败: {e}")
