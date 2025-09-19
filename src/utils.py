# -*- coding: utf-8 -*-
"""
辅助工具模块 (Utilities)

本模块包含一系列独立的辅助类和函数，用于支持图数据构建流程。
主要包括：
- SolarCalculator: 根据地理位置和时间计算太阳方位角和高度角。
- CSVWeatherParser: 解析EPW格式的气象数据CSV文件。
- compute_global_maxes: 在所有输入数据中计算全局的最大树高和建筑高度，用于归一化。
- Reporting Functions: 用于生成和打印图数据结构报告的函数，方便调试和验证。
"""
import numpy as np
import pandas as pd
import pvlib
import math
import networkx as nx
from datetime import datetime, timedelta
from pathlib import Path

# PyTorch Geometric imports for reporting functions
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# ==================== 太阳位置计算模块 ====================
class SolarCalculator:
    """根据给定的地理坐标和时间计算太阳的位置。"""
    def __init__(self, latitude: float, longitude: float, altitude: float = 0, timezone_str: str = 'Asia/Singapore'):
        """
        初始化太阳位置计算器。

        Args:
            latitude (float): 纬度。
            longitude (float): 经度。
            altitude (float): 海拔高度 (米)。
            timezone_str (str): 时区字符串 (例如 'Asia/Singapore')。
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone_str = timezone_str

    def get_solar_position(self, year: int, month: int, day: int, hour_of_day: int) -> dict:
        """
        获取指定日期和小时的太阳位置。

        Args:
            year (int): 年份。
            month (int): 月份。
            day (int): 日期。
            hour_of_day (int): 小时 (0-23)。

        Returns:
            dict: 包含太阳方位角和高度角的字典 (度和弧度)。
        """
        dt_local = datetime(year, month, day, hour_of_day, 0, 0)
        dt_aware_local = pd.Timestamp(dt_local, tz=self.timezone_str)
        times = pd.DatetimeIndex([dt_aware_local])

        solar_position = pvlib.solarposition.get_solarposition(
            time=times,
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            temperature=25,  # 平均温度，对太阳位置影响不大
            pressure=pvlib.atmosphere.alt2pres(self.altitude)
        )
        azimuth_deg = solar_position['azimuth'].iloc[0]
        elevation_deg = solar_position['elevation'].iloc[0]

        return {
            'azimuth_deg': azimuth_deg,
            'elevation_deg': elevation_deg,
            'azimuth_rad': math.radians(azimuth_deg),
            'elevation_rad': math.radians(elevation_deg)
        }

# ==================== CSV气象数据处理模块 ===================
class CSVWeatherParser:
    """解析EPW气象数据CSV文件，并按小时提供数据。"""
    def __init__(self, csv_path: str):
        """
        初始化气象数据解析器。

        Args:
            csv_path (str): CSV文件的路径。

        Raises:
            ValueError: 如果文件无法读取或缺少必要的列。
        """
        self.required_columns = [
            'Dry Bulb Temperature {C}', 'Relative Humidity {%}',
            'Wind Speed {m/s}', 'Atmospheric Pressure {Pa}',
            'Global Horizontal Radiation {Wh/m2}', 'Wind Direction {deg}'
        ]
        self.df = self._load_and_preprocess_csv(csv_path)

    def _load_and_preprocess_csv(self, path: str) -> pd.DataFrame:
        """加载并预处理CSV文件。"""
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"无法读取CSV文件 '{path}': {e}")

        expected_datetime_cols = ['Date', 'HH:MM']
        for col in expected_datetime_cols:
            if col not in df.columns:
                raise ValueError(f"CSV文件 '{path}' 缺少必要的日期/时间列: '{col}'")
        try:
            df['datetime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['HH:MM'].astype(str),
                format='%Y/%m/%d %H:%M', errors='raise'
            )
        except Exception as e:
            raise ValueError(f"解析日期时间列时出错: {e}.")

        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour

        for col_name in self.required_columns:
            if col_name not in df.columns:
                raise ValueError(f"CSV文件 '{path}' 缺少必要的气象数据列: '{col_name}'")
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        return df

    def get_hourly_data(self, target_month: int, target_day: int, target_hour: int) -> dict:
        """
        获取指定小时的气象数据。如果精确数据不存在，则进行回退填充。

        Args:
            target_month (int): 目标月份。
            target_day (int): 目标日期。
            target_hour (int): 目标小时。

        Returns:
            dict: 包含该小时所有气象参数的字典。
        """
        mask = (
            (self.df['month'] == target_month) &
            (self.df['day'] == target_day) &
            (self.df['hour'] == target_hour)
        )
        selected_data = self.df[mask]

        if selected_data.empty:
            fallback_mask = (self.df['month'] == target_month) & (self.df['hour'] == target_hour)
            fallback_data = self.df[fallback_mask]
            if not fallback_data.empty:
                print(f"警告: 未找到 {target_month}-{target_day} {target_hour}:00 的精确气象数据。将使用该月该小时的首条记录。")
                selected_data = fallback_data.iloc[[0]]
            else:
                raise ValueError(f"未找到气象数据：月份={target_month}, 日期={target_day}, 小时={target_hour}, 且无回退数据。")

        row = selected_data.iloc[0].copy()
        for col_name in self.required_columns:
            if pd.isna(row[col_name]):
                mean_val = self.df[(self.df['month'] == target_month) & (self.df['hour'] == target_hour)][col_name].mean()
                if pd.notna(mean_val):
                    print(f"警告: 在 {target_month}-{target_day} {target_hour}:00 的数据中，列 '{col_name}' 无效(NaN)。已用月均值 {mean_val:.2f} 填充。")
                    row[col_name] = mean_val
                else:
                    raise ValueError(f"在 {target_month}-{target_day} {target_hour}:00 的数据中，列 '{col_name}' 包含无效值 (NaN) 且无法用月均值填充。")

        wind_direction_deg_raw = row['Wind Direction {deg}']
        wind_blowing_to_meteo_deg = (wind_direction_deg_raw + 180) % 360

        return {
            'temperature_c': row['Dry Bulb Temperature {C}'],
            'humidity_percent': row['Relative Humidity {%}'],
            'wind_speed_mps': row['Wind Speed {m/s}'],
            'pressure_pa': row['Atmospheric Pressure {Pa}'],
            'global_horizontal_radiation_whm2': row['Global Horizontal Radiation {Wh/m2}'],
            'wind_direction_from_deg': wind_direction_deg_raw,
            'wind_blowing_to_meteo_deg': wind_blowing_to_meteo_deg,
            'wind_blowing_to_meteo_rad': math.radians(wind_blowing_to_meteo_deg)
        }

# ==================== 全局参数计算模块 ===================
def compute_global_maxes(input_dir: str) -> tuple[float, float]:
    """
    遍历所有输入NPY文件，计算全局最大树高和建筑高度。

    Args:
        input_dir (str): 包含`Input_*.npy`文件的目录。

    Returns:
        tuple[float, float]: (全局最大树高, 全局最大建筑高度)。
    """
    input_files = sorted(Path(input_dir).glob("Input_*.npy"))
    if not input_files: return 1.0, 1.0

    global_tree_max = -np.inf
    global_building_max = -np.inf

    for file_path in input_files:
        data = np.load(file_path).astype(np.float32)
        if data.ndim == 3 and data.shape[2] >= 3:
            global_tree_max = max(global_tree_max, data[:, :, 0].max())
            global_building_max = max(global_building_max, data[:, :, 2].max())

    final_tree_max = global_tree_max if np.isfinite(global_tree_max) and global_tree_max > 0 else 1.0
    final_bldg_max = global_building_max if np.isfinite(global_building_max) and global_building_max > 0 else 1.0
    return final_tree_max, final_bldg_max


# ==================== 报告和验证函数 ===================

def generate_data_report(graph: Data):
    """为单个图对象生成详细的结构和内容报告。"""
    report = ["="*40 + "\n图数据结构分析报告 (单小时)\n" + "="*40]
    report.append(f"\n[文件ID (如有)]: {getattr(graph, 'file_id', 'N/A')}")
    report.append(f"[小时 (如有)]: {getattr(graph, 'hour_of_day', 'N/A')}")
    report.append(f"\n[维度信息]")
    report.append(f"节点数量: {graph.num_nodes}")
    report.append(f"边数量: {graph.num_edges}")
    report.append(f"节点特征维度: {graph.x.shape}")
    
    edge_attr_shape_str = str(graph.edge_attr.shape) if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else 'N/A'
    report.append(f"边特征维度 (edge_attr): {edge_attr_shape_str}")
    report.append(f"边权重维度 (edge_weight): {graph.edge_weight.shape if hasattr(graph, 'edge_weight') and graph.edge_weight is not None else 'N/A'}")
    report.append(f"目标值维度: {graph.y.shape}")

    report.append("\n[节点特征X (前5行样本)]:\n" + str(graph.x[:5].numpy()))
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None and graph.num_edges > 0:
        report.append("\n[边特征EA (前5行样本)]:\n" + str(graph.edge_attr[:5].numpy()))
    if hasattr(graph, 'edge_weight') and graph.edge_weight is not None and graph.num_edges > 0:
        report.append("\n[独立边权重EW (前5行样本)]:\n" + str(graph.edge_weight[:5].numpy()))
        report.append(f"  独立边权重统计: min={graph.edge_weight.min().item():.3f}, max={graph.edge_weight.max().item():.3f}, mean={graph.edge_weight.mean().item():.3f}")

    report.append("\n[目标值Y (前5行样本)]:\n" + str(graph.y[:5].numpy()))

    if hasattr(graph, 'graph_global_env_features') and graph.graph_global_env_features is not None:
        report.append("\n[图级别全局环境特征 (7维)]:\n" + str(graph.graph_global_env_features.numpy()))

    final_report = "\n".join(report)
    print(final_report)

def verify_edge_structure(graph: Data, grid_size: int = 4, sample_size: int = 10):
    """抽样验证图中边的属性（距离、dx、dy）是否与其节点位置一致。"""
    if not hasattr(graph, 'edge_attr') or graph.edge_attr is None or graph.edge_attr.size(0) == 0 or graph.edge_attr.size(1) < 3:
        print("[verify_edge_structure] 警告：edge_attr 列数不足 (<3) 或无边，无法验证。")
        return

    edge_attrs_np = graph.edge_attr.cpu().numpy()
    edge_index_np = graph.edge_index.cpu().numpy()
    pos_np = graph.pos.cpu().numpy()

    num_edges_to_sample = min(sample_size, graph.num_edges)
    if num_edges_to_sample == 0:
        print("    无边可供抽样检查。")
        return

    sampled_eids = np.random.choice(graph.num_edges, num_edges_to_sample, replace=False)
    print(f"\n    抽样 {num_edges_to_sample} 条边进行 (dist, dx, dy) 对比:")
    for i, eid in enumerate(sampled_eids, 1):
        src_idx, dst_idx = edge_index_np[0, eid], edge_index_np[1, eid]
        dist_stored, dx_grid_stored, dy_grid_stored = edge_attrs_np[eid, 0], edge_attrs_np[eid, 1], edge_attrs_np[eid, 2]
        
        pos_src, pos_dst = pos_np[src_idx], pos_np[dst_idx]
        dist_calc = np.linalg.norm(pos_dst - pos_src)
        dx_grid_calc = (pos_dst[0] - pos_src[0]) / grid_size
        dy_grid_calc = (pos_dst[1] - pos_src[1]) / grid_size
        
        print(f"    [{i}] EdgeID={eid} ({src_idx}->{dst_idx})")
        print(f"      存储: dist={dist_stored:.2f}, dx_g={dx_grid_stored:.2f}, dy_g={dy_grid_stored:.2f}")
        print(f"      计算: dist={dist_calc:.2f}, dx_g={dx_grid_calc:.2f}, dy_g={dy_grid_calc:.2f}")
        if abs(dist_stored-dist_calc) > 1e-1 or abs(dx_grid_stored-dx_grid_calc) > 0.6 or abs(dy_grid_stored-dy_grid_calc) > 0.6 :
            print(f"      [警告] 差异较大!")
        else:
            print(f"      [OK]")

def generate_sequence_y_report(graph_sequence: list, sequence_index: int = 0):
    """为单个图序列中的每个图的目标值(y)生成详细报告。"""
    if not graph_sequence:
        print(f"序列 {sequence_index} 为空，无法生成y值报告。")
        return

    report_lines = [
        f"\n{'='*25} Y 值详细报告: 序列 {sequence_index} {'='*25}",
        f"序列中图的数量 (小时数): {len(graph_sequence)}"
    ]

    for i, graph_data in enumerate(graph_sequence):
        if graph_data is None:
            report_lines.append(f"\n--- [序列内索引 {i}] 图数据: None ---")
            continue

        actual_hour = getattr(graph_data, 'hour_of_day', 'N/A')
        report_lines.append(f"\n--- [序列内索引 {i}] 实际小时: {actual_hour if actual_hour != 'N/A' else '(未知)'}:00 ---")

        y_tensor = graph_data.y
        if y_tensor is None:
            report_lines.append("  y 值: None")
        else:
            y_np = y_tensor.cpu().numpy().flatten()
            nan_count = np.sum(np.isnan(y_np))
            non_nan_y = y_np[~np.isnan(y_np)]

            report_lines.append(f"  y 形状: {list(y_tensor.shape)}, NaN数量: {nan_count}")

            if non_nan_y.size > 0:
                report_lines.append(f"  y 值统计 (非NaN): Min={np.min(non_nan_y):.4f}, Max={np.max(non_nan_y):.4f}, Mean={np.mean(non_nan_y):.4f}")
            else:
                report_lines.append("  y 值统计: 所有值均为 NaN 或为空。")
    
    print("\n".join(report_lines))


def generate_time_features_for_sequence(base_dt_obj: datetime, num_steps: int) -> torch.Tensor:
    """
    为时间序列生成周期性时间特征。

    Args:
        base_dt_obj (datetime): 序列的起始时间。
        num_steps (int): 要生成的时间步数。

    Returns:
        torch.Tensor: 形状为 (num_steps, 4) 的时间特征张量。
    """
    features_list = []
    for i in range(num_steps):
        current_dt = base_dt_obj + timedelta(hours=i)
        hour_norm = current_dt.hour / 23.0
        yday = current_dt.timetuple().tm_yday
        days_in_year = 366.0 if current_dt.year % 4 == 0 and (
                    current_dt.year % 100 != 0 or current_dt.year % 400 == 0) else 365.0
        doy_norm = yday / days_in_year

        hour_sin = math.sin(2 * math.pi * hour_norm)
        hour_cos = math.cos(2 * math.pi * hour_norm)
        doy_sin = math.sin(2 * math.pi * doy_norm)
        doy_cos = math.cos(2 * math.pi * doy_norm)
        features_list.append(torch.tensor([hour_sin, hour_cos, doy_sin, doy_cos], dtype=torch.float32))
    return torch.stack(features_list)


def calculate_aggregated_metrics_report(hourly_metrics: dict, T_pred_horizon: int) -> dict:
    """从每小时指标字典中计算平均值和标准差。"""
    aggregated = {}
    for metric in ['r2', 'mse', 'mae', 'rmse']:
        values = [hourly_metrics[t][metric] for t in range(T_pred_horizon) if
                  t in hourly_metrics and not np.isnan(hourly_metrics[t][metric])]
        if values:
            aggregated[f'avg_{metric}'] = np.mean(values)
            aggregated[f'std_{metric}'] = np.std(values)
        else:
            aggregated[f'avg_{metric}'] = np.nan
            aggregated[f'std_{metric}'] = np.nan
    return aggregated


def print_hourly_metrics_summary(set_name: str, hourly_metrics: dict, T_pred_horizon: int):
    """格式化打印每小时和聚合的评估指标。"""
    if not hourly_metrics:
        print(f"    {set_name} metrics not available.")
        return

    print(f"\n    --- {set_name} Metrics (Original Scale) ---")
    print(f"    {'Hour':<5} | {'R2':>10s} | {'RMSE':>12s} | {'MAE':>12s} | {'Count':>7s}")
    print(f"    {'-' * 5} | {'-' * 10} | {'-' * 12} | {'-' * 12} | {'-' * 7}")
    for hour in range(T_pred_horizon):
        metrics = hourly_metrics.get(hour, {})
        print(
            f"    {hour:<5} | {metrics.get('r2', np.nan):10.4f} | {metrics.get('rmse', np.nan):12.4f} | {metrics.get('mae', np.nan):12.4f} | {metrics.get('count', 0):7d}")

    agg = calculate_aggregated_metrics_report(hourly_metrics, T_pred_horizon)
    print(f"    -------------------------------------------------------")
    print(f"    Avg R2  : {agg.get('avg_r2', np.nan):.4f} (Std: {agg.get('std_r2', np.nan):.4f})")
    print(f"    Avg RMSE: {agg.get('avg_rmse', np.nan):.4f} (Std: {agg.get('std_rmse', np.nan):.4f})")
    print(f"    Avg MAE : {agg.get('avg_mae', np.nan):.4f} (Std: {agg.get('std_mae', np.nan):.4f})")
