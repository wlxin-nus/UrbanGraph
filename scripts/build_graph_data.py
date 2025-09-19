# -*- coding: utf-8 -*-
"""
主执行脚本：构建时空图序列数据

本脚本是整个数据处理流程的入口。它负责：
1.  解析命令行参数，获取输入/输出路径和配置。
2.  调用 `src.utils` 中的函数计算全局最大值。
3.  遍历所有原始数据文件对 (Input/Output)。
4.  对每个文件，使用 `src.data_processing.ClimateDataPreprocessor` 进行预处理和窗口化。
5.  对每个窗口，遍历12个小时，使用 `src.graph_construction.GraphConstructor` 构建逐小时的图。
6.  (可选) 使用 `src.graph_construction.GraphAugmentor` 对图进行增广。
7.  将所有窗口生成的12小时图序列聚合在一起。
8.  将最终的数据集（列表的列表）保存为 pickle 文件。
"""
import argparse
import pickle
import re
import traceback
from pathlib import Path

# 从 src 模块中导入必要的类和函数
from src.utils import SolarCalculator, CSVWeatherParser, compute_global_maxes, generate_data_report, verify_edge_structure, generate_sequence_y_report
from src.data_processing import ClimateDataPreprocessor
from src.graph_construction import GraphConstructor, GraphAugmentor

def process_sequences(args):
    """根据命令行参数执行完整的数据处理流程。"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_files = sorted(input_dir.glob("Input_*.npy"))

    if not input_files:
        print(f"错误: 在 '{input_dir}' 未找到任何 Input_*.npy 文件。")
        return []

    # --- 1. 全局初始化 ---
    global_tree_max, global_building_max = compute_global_maxes(str(input_dir))
    print(f"全局最大树高: {global_tree_max:.2f}, 全局最大建筑高度: {global_building_max:.2f}")

    solar_calc = SolarCalculator(latitude=1.3521, longitude=103.8198, timezone_str='Asia/Singapore')
    weather_parser = CSVWeatherParser(csv_path=args.weather_csv)
    
    gc_params = vars(args) # 将argparse的参数转为字典
    gc_params['global_tree_max_val'] = global_tree_max
    gc_params['global_building_max_val'] = global_building_max
    graph_builder = GraphConstructor(**gc_params)
    
    augmentor = GraphAugmentor(add_neighbor_agg=args.add_neighbor_agg) if args.add_neighbor_agg else None

    all_window_sequences = []

    # --- 2. 遍历文件 ---
    for input_path in input_files:
        match = re.search(r"Input_(\d+).npy", input_path.name)
        if not match: continue
        file_id = match.group(1)
        output_path_npy = output_dir / f"Output_{file_id}.npy"

        if not output_path_npy.exists():
            print(f"跳过 {input_path.name}: 缺少对应 Output_{file_id}.npy")
            continue

        print(f"\n=== 开始处理文件对 {file_id} ===")
        try:
            # --- 3. 数据预处理和窗口化 ---
            preprocessor = ClimateDataPreprocessor(
                str(input_path), str(output_path_npy), args.window_size, args.stride,
                global_tree_max, global_building_max
            )
            input_windows, output_windows = preprocessor.process()
            hours_in_npy = preprocessor.output_data.shape[3]

            # --- 4. 遍历窗口，构建图序列 ---
            for win_idx, (inp_win, out_win) in enumerate(zip(input_windows, output_windows)):
                current_window_graphs = []
                # print(f"  -- 处理文件 {file_id}, 窗口索引 {win_idx} --")
                
                # --- 5. 遍历小时 ---
                for hour_offset in range(args.num_hours_in_sequence):
                    actual_hour = args.start_hour_of_day + hour_offset
                    try:
                        solar_params = solar_calc.get_solar_position(args.year, args.month, args.day, actual_hour)
                        weather_params = weather_parser.get_hourly_data(args.month, args.day, actual_hour)
                        
                        y_slice_idx = actual_hour - args.output_npy_start_hour
                        
                        # --- 6. 构建图 ---
                        graph = graph_builder.build_graph(
                            inp_win, out_win, weather_params, solar_params, y_slice_idx
                        )
                        graph.file_id, graph.window_index = file_id, win_idx
                        graph.hour_of_day, graph.hour_index_in_sequence = actual_hour, hour_offset

                        if not (0 <= y_slice_idx < hours_in_npy):
                            graph.y.fill_(args.output_npy_fill_value)
                        
                        if augmentor:
                            graph = augmentor.augment_static(graph)
                        
                        current_window_graphs.append(graph)
                    
                    except Exception as e:
                        print(f"    构建图失败 (文件:{file_id}, 窗口:{win_idx}, 小时:{actual_hour}): {e}")
                        traceback.print_exc()
                        break # 如果一个小时失败，则跳过整个窗口

                if len(current_window_graphs) == args.num_hours_in_sequence:
                    all_window_sequences.append(current_window_graphs)

        except Exception as e:
            print(f"处理文件对 {file_id} 时发生严重错误: {e}")
            traceback.print_exc()
            continue

    print(f"\n处理完成，共生成 {len(all_window_sequences)} 个窗口的图序列。")
    return all_window_sequences

def main():
    parser = argparse.ArgumentParser(description="从原始NPY和CSV数据构建时空图序列。")

    # --- 路径参数 ---
    parser.add_argument('--base_dir', type=str, default="data", help='存放Input, Output和CSV的数据根目录')
    parser.add_argument('--save_dir', type=str, default="data/processed", help='保存处理后pickle文件的目录')

    # --- 时间参数 ---
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--month', type=int, default=5)
    parser.add_argument('--day', type=int, default=3)
    parser.add_argument('--start_hour_of_day', type=int, default=7, help='图序列的起始小时 (e.g., 7 for 7AM)')
    parser.add_argument('--num_hours_in_sequence', type=int, default=13, help='每个序列包含的小时数 (e.g., 13 for 7AM-7PM)')
    parser.add_argument('--output_npy_start_hour', type=int, default=8, help='Output.npy文件中第0个时间片对应的实际小时')
    parser.add_argument('--output_npy_fill_value', type=float, default=0.0, help='当目标(y)数据缺失时使用的填充值 (e.g., 0.0 or nan)')

    # --- 数据处理参数 ---
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--stride', type=int, default=40)
    parser.add_argument('--target_attr_index', type=int, default=5, help='目标变量在Output.npy中的索引')
    
    # --- 图构建参数 (GraphConstructor) ---
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--k_similarity', type=int, default=8)
    parser.add_argument('--base_building_shadow_radius_max_grids', type=int, default=15)
    parser.add_argument('--base_tree_shadow_radius_max_grids', type=int, default=5)
    parser.add_argument('--base_tree_activity_radius_max_grids', type=int, default=5)
    parser.add_argument('--distance_decay_factor_per_grid', type=float, default=0.01)
    parser.add_argument('--similarity_dist_decay_factor_per_grid', type=float, default=0.005)
    parser.add_argument('--actual_shadow_boost_factor', type=float, default=1.2)
    parser.add_argument('--tree_activity_height_influence_factor', type=float, default=0.2)
    parser.add_argument('--shadow_angular_width_deg', type=float, default=25.0)

    # --- 图增广参数 (GraphAugmentor) ---
    parser.add_argument('--add_neighbor_agg', action='store_true', help='是否添加邻居节点特征聚合')
    
    args = parser.parse_args()

    # 动态设置输入/输出路径
    args.input_dir = Path(args.base_dir) / "Input"
    args.output_dir = Path(args.base_dir) / "Output"
    args.weather_csv = Path(args.base_dir) / "SGP_SINGAPORE-CHANGI-IAP_486980S_23EPW.csv"
    
    # --- 开始处理 ---
    all_sequences = process_sequences(args)

    # --- 保存结果 ---
    if all_sequences:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_name = f"graph_seq_{args.year}{args.month:02d}{args.day:02d}.pkl"
        save_path = save_dir / save_name
        
        with open(save_path, "wb") as f:
            pickle.dump(all_sequences, f)
        print(f"\n[已保存] 图序列数据 => {save_path}")

        # --- 报告 ---
        print("\n=== 抽样检查首个窗口的首个图 ===")
        if all_sequences[0] and all_sequences[0][0]:
            first_graph = all_sequences[0][0]
            generate_data_report(first_graph)
            verify_edge_structure(first_graph, grid_size=args.grid_size)

        print("\n=== Y值详细报告 (首个窗口序列) ===")
        if all_sequences[0]:
            generate_sequence_y_report(all_sequences[0], sequence_index=0)
    else:
        print("\n未生成任何图序列数据。")

if __name__ == "__main__":
    main()
