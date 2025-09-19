# -*- coding: utf-8 -*-
"""
数据预处理模块 (Data Preprocessing)

本模块负责处理原始的.npy格式的气候模拟数据。
主要功能包括：
- ClimateDataPreprocessor:
  - 加载输入的静态地理特征 (Input_*.npy) 和输出的动态气候变量 (Output_*.npy)。
  - 对输出数据中的NaN值进行插值处理。
  - 将大的250x250网格数据切分为指定大小和步长的滑动窗口。
  - 将输入的地理特征转换为模型可用的节点特征 (例如，对表面类型进行one-hot编码)。
"""
import numpy as np

class ClimateDataPreprocessor:
    """处理原始NPY数据，包括NaN填充、特征工程和窗口化。"""
    def __init__(self, input_path: str, output_path: str, window_size: int = 50, stride: int = 40,
                 global_tree_max: float = None, global_building_max: float = None):
        """
        初始化数据预处理器。

        Args:
            input_path (str): 输入NPY文件路径 (静态特征)。
            output_path (str): 输出NPY文件路径 (动态目标变量)。
            window_size (int): 滑动窗口的边长。
            stride (int): 滑动窗口的步长。
            global_tree_max (float): 用于归一化树高的全局最大值。
            global_building_max (float): 用于归一化建筑高度的全局最大值。
        """
        self.input_data = np.load(input_path).astype(np.float32)
        raw_output = np.load(output_path).astype(np.float32)
        self.window_size = window_size
        self.stride = stride
        self.building_mask_full = (self.input_data[:, :, 2] > 0)

        self.global_tree_max = global_tree_max if global_tree_max is not None else self.input_data[:, :, 0].max()
        if self.global_tree_max == 0: self.global_tree_max = 1.0
        self.global_building_max = global_building_max if global_building_max is not None else self.input_data[:, :, 2].max()
        if self.global_building_max == 0: self.global_building_max = 1.0

        self.output_data = self._process_nan(raw_output)
        self._validate_shapes()

    def _validate_shapes(self):
        """验证输入和输出数据的形状是否符合预期。"""
        input_shape = self.input_data.shape
        output_shape = self.output_data.shape
        assert len(input_shape) == 3 and input_shape[:2] == (250, 250), \
            f"输入数据形状应为(250,250,C)，实际得到{input_shape}"
        assert len(output_shape) == 4 and output_shape[1:3] == (250, 250), \
            f"输出数据形状应为(V,250,250,T)，实际得到{output_shape}"

    def _process_nan(self, raw_output: np.ndarray) -> np.ndarray:
        """使用非建筑区域的均值填充该区域的NaN值。"""
        building_mask_expanded = self.building_mask_full[np.newaxis, :, :, np.newaxis]
        cleaned = raw_output.copy()
        
        for var_idx in range(cleaned.shape[0]):
            for t_idx in range(cleaned.shape[3]):
                slice_data = cleaned[var_idx, :, :, t_idx]
                mask_non_building = ~self.building_mask_full
                valid_mean_non_building = np.nanmean(slice_data[mask_non_building])
                if np.isnan(valid_mean_non_building):
                    valid_mean_non_building = 0 # Fallback if all non-building are NaNs
                
                slice_data[np.isnan(slice_data) & mask_non_building] = valid_mean_non_building
                cleaned[var_idx, :, :, t_idx] = slice_data
        
        return cleaned

    def _generate_windows(self, data: np.ndarray, is_output: bool = False) -> np.ndarray:
        """根据设定的窗口大小和步长，从大数据块中切分出小窗口。"""
        h, w = data.shape[1:3] if is_output else data.shape[:2]
        windows = []
        for i in range(0, h - self.window_size + 1, self.stride):
            for j in range(0, w - self.window_size + 1, self.stride):
                if is_output:
                    window = data[:, i:i+self.window_size, j:j+self.window_size, :]
                else:
                    window = data[i:i+self.window_size, j:j+self.window_size]
                windows.append(window)
        return np.array(windows)

    def process(self) -> tuple[np.ndarray, np.ndarray]:
        """执行完整的预处理流程。"""
        input_features = self._create_input_features()
        input_windows = self._generate_windows(input_features)
        output_windows = self._generate_windows(self.output_data, is_output=True)
        return input_windows, output_windows

    def _create_input_features(self) -> np.ndarray:
        """从原始输入数据创建节点特征。"""
        surface = self.input_data[:, :, 1].astype(int)
        surface_clipped = np.clip(surface, 1, 5)
        onehot = np.eye(5)[surface_clipped - 1]
        
        tree_norm = self.input_data[:, :, 0] / self.global_tree_max
        building_norm = self.input_data[:, :, 2] / self.global_building_max
        building_mask_float = self.building_mask_full.astype(float)

        return np.concatenate([
            tree_norm[..., None],
            onehot,
            building_norm[..., None],
            building_mask_float[..., None]
        ], axis=-1)
