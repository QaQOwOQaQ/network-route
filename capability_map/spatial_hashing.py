"""
空间网格索引/分区建模算法
使用空间哈希和分区方法构建能力底图
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .scenario_generator import CapabilityNode, SpatialCell


class SpatialHashingModeling:
    """空间网格索引/分区建模"""
    
    def __init__(self, nodes: List[CapabilityNode], spatial_grid: List[SpatialCell]):
        """
        初始化空间哈希建模
        
        Args:
            nodes: 节点列表
            spatial_grid: 空间网格单元列表
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.spatial_grid = spatial_grid
        self.grid_dict = {cell.cell_id: cell for cell in spatial_grid}
        self.capability_map = {}  # 能力底图缓存
    
    def hash_position(self, position: Tuple[float, float], grid_size: Tuple[int, int], area_size: Tuple[float, float]) -> int:
        """
        将位置哈希到网格单元
        
        Args:
            position: 位置 (x, y)
            area_size: 区域大小
            grid_size: 网格大小
            
        Returns:
            网格单元ID
        """
        x, y = position
        cell_width = area_size[0] / grid_size[0]
        cell_height = area_size[1] / grid_size[1]
        
        col = int(x / cell_width)
        row = int(y / cell_height)
        
        # 确保在范围内
        col = max(0, min(col, grid_size[0] - 1))
        row = max(0, min(row, grid_size[1] - 1))
        
        return row * grid_size[0] + col
    
    def query_capability_in_region(
        self,
        region_bounds: Tuple[float, float, float, float],
        grid_size: Tuple[int, int],
        area_size: Tuple[float, float]
    ) -> Dict:
        """
        查询区域内的能力
        
        Args:
            region_bounds: 区域边界 (x_min, y_min, x_max, y_max)
            grid_size: 网格大小
            area_size: 区域大小
            
        Returns:
            能力信息字典
        """
        x_min, y_min, x_max, y_max = region_bounds
        
        # 找出覆盖该区域的所有网格单元
        cell_width = area_size[0] / grid_size[0]
        cell_height = area_size[1] / grid_size[1]
        
        col_min = int(x_min / cell_width)
        col_max = int(x_max / cell_width) + 1
        row_min = int(y_min / cell_height)
        row_max = int(y_max / cell_height) + 1
        
        # 确保在范围内
        col_min = max(0, col_min)
        col_max = min(col_max, grid_size[0])
        row_min = max(0, row_min)
        row_max = min(row_max, grid_size[1])
        
        total_capability = 0.0
        node_count = 0
        nodes_in_region = []
        
        for row in range(row_min, row_max):
            for col in range(col_min, col_max):
                cell_id = row * grid_size[0] + col
                if cell_id in self.grid_dict:
                    cell = self.grid_dict[cell_id]
                    # 检查节点是否在查询区域内
                    for node_id in cell.nodes:
                        node = self.nodes[node_id]
                        x, y = node.position
                        if x_min <= x < x_max and y_min <= y < y_max:
                            total_capability += node.capability
                            node_count += 1
                            nodes_in_region.append(node_id)
        
        return {
            'total_capability': total_capability,
            'node_count': node_count,
            'nodes': nodes_in_region,
            'average_capability': total_capability / node_count if node_count > 0 else 0.0
        }
    
    def build_capability_map(
        self,
        grid_size: Tuple[int, int],
        area_size: Tuple[float, float],
        time: Optional[float] = None
    ) -> Dict[int, float]:
        """
        构建能力底图
        
        Args:
            grid_size: 网格大小
            area_size: 区域大小
            time: 时间点（如果考虑时间窗口）
            
        Returns:
            {网格单元ID: 总能力值}
        """
        capability_map = {}
        
        for cell in self.spatial_grid:
            total_capability = 0.0
            for node_id in cell.nodes:
                node = self.nodes[node_id]
                # 如果指定了时间，检查节点是否可用
                if time is None or (node.available_time <= time <= node.unavailable_time):
                    total_capability += node.capability
            
            capability_map[cell.cell_id] = total_capability
        
        self.capability_map = capability_map
        return capability_map
    
    def find_nearest_capable_nodes(
        self,
        position: Tuple[float, float],
        required_capability: float,
        grid_size: Tuple[int, int],
        area_size: Tuple[float, float],
        max_distance: float = 500.0
    ) -> List[Tuple[int, float, float]]:
        """
        查找最近的有能力节点
        
        Args:
            position: 查询位置
            required_capability: 所需能力
            grid_size: 网格大小
            area_size: 区域大小
            max_distance: 最大搜索距离
            
        Returns:
            [(节点ID, 距离, 能力值), ...]
        """
        # 先找到所在网格单元
        cell_id = self.hash_position(position, grid_size, area_size)
        
        # 搜索相邻网格单元
        cell_width = area_size[0] / grid_size[0]
        cell_height = area_size[1] / grid_size[1]
        
        search_radius = int(max_distance / min(cell_width, cell_height)) + 1
        
        col = cell_id % grid_size[0]
        row = cell_id // grid_size[0]
        
        candidates = []
        
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                new_row = row + dr
                new_col = col + dc
                
                if 0 <= new_row < grid_size[1] and 0 <= new_col < grid_size[0]:
                    neighbor_cell_id = new_row * grid_size[0] + new_col
                    if neighbor_cell_id in self.grid_dict:
                        cell = self.grid_dict[neighbor_cell_id]
                        for node_id in cell.nodes:
                            node = self.nodes[node_id]
                            if node.capability >= required_capability:
                                distance = np.sqrt(
                                    (node.position[0] - position[0])**2 +
                                    (node.position[1] - position[1])**2
                                )
                                if distance <= max_distance:
                                    candidates.append((node_id, distance, node.capability))
        
        # 按距离排序
        candidates.sort(key=lambda x: x[1])
        return candidates
    
    def partition_network(
        self,
        num_partitions: int,
        grid_size: Tuple[int, int]
    ) -> Dict[int, List[int]]:
        """
        对网络进行分区
        
        Args:
            num_partitions: 分区数量
            grid_size: 网格大小
            
        Returns:
            {分区ID: [网格单元ID列表]}
        """
        partitions = {i: [] for i in range(num_partitions)}
        
        # 简单的网格分区策略：按行或列划分
        cells_per_partition = len(self.spatial_grid) // num_partitions
        
        for i, cell in enumerate(self.spatial_grid):
            partition_id = min(i // cells_per_partition, num_partitions - 1)
            partitions[partition_id].append(cell.cell_id)
        
        return partitions

