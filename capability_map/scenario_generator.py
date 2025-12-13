"""
能力底图构建场景生成模块
生成网络能力、空间分布、时间演化等数据
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class CapabilityNode:
    """能力节点"""
    node_id: int
    position: Tuple[float, float]  # (x, y) 坐标
    capability: float  # 节点能力值
    available_time: float  # 可用时间开始
    unavailable_time: float  # 可用时间结束
    contact_windows: List[Tuple[float, float]]  # 接触时间窗口列表


@dataclass
class Contact:
    """接触（连接）"""
    source: int
    target: int
    start_time: float  # 接触开始时间
    end_time: float  # 接触结束时间
    bandwidth: float  # 带宽
    distance: float  # 距离


@dataclass
class SpatialCell:
    """空间网格单元"""
    cell_id: int
    bounds: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    nodes: List[int]  # 包含的节点ID列表
    total_capability: float  # 总能力值


class CapabilityMapScenarioGenerator:
    """能力底图构建场景生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化场景生成器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_network_topology(
        self,
        num_nodes: int = 30,
        area_size: Tuple[float, float] = (1000, 1000),
        time_horizon: float = 200.0
    ) -> Tuple[List[CapabilityNode], List[Contact], nx.Graph]:
        """
        生成网络拓扑（用于TEG/CGR建模）
        
        Args:
            num_nodes: 节点数量
            area_size: 区域大小
            time_horizon: 时间范围
            
        Returns:
            (节点列表, 接触列表, NetworkX图)
        """
        nodes = []
        contacts = []
        G = nx.Graph()
        
        # 生成节点
        for i in range(num_nodes):
            x = np.random.uniform(0, area_size[0])
            y = np.random.uniform(0, area_size[1])
            capability = np.random.uniform(50, 200)
            
            # 生成可用时间窗口
            available_time = np.random.uniform(0, time_horizon * 0.3)
            unavailable_time = available_time + np.random.uniform(time_horizon * 0.2, time_horizon * 0.5)
            
            # 生成接触时间窗口
            num_contacts = np.random.randint(1, 4)
            contact_windows = []
            for _ in range(num_contacts):
                start = np.random.uniform(0, time_horizon * 0.8)
                end = start + np.random.uniform(10, time_horizon * 0.2)
                contact_windows.append((start, end))
            
            node = CapabilityNode(
                node_id=i,
                position=(x, y),
                capability=capability,
                available_time=available_time,
                unavailable_time=unavailable_time,
                contact_windows=contact_windows
            )
            nodes.append(node)
            G.add_node(i, pos=(x, y), capability=capability)
        
        # 生成接触（基于距离和概率）
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                pos_i = nodes[i].position
                pos_j = nodes[j].position
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                
                # 距离越近，接触概率越高
                max_distance = np.sqrt(area_size[0]**2 + area_size[1]**2)
                prob = 0.3 * (1 - distance / max_distance)
                
                if np.random.random() < prob:
                    start_time = np.random.uniform(0, time_horizon * 0.7)
                    end_time = start_time + np.random.uniform(20, time_horizon * 0.3)
                    bandwidth = np.random.uniform(10, 100)
                    
                    contact = Contact(
                        source=i,
                        target=j,
                        start_time=start_time,
                        end_time=end_time,
                        bandwidth=bandwidth,
                        distance=distance
                    )
                    contacts.append(contact)
                    G.add_edge(i, j, 
                             start_time=start_time,
                             end_time=end_time,
                             bandwidth=bandwidth,
                             distance=distance)
        
        return nodes, contacts, G
    
    def generate_spatial_grid(
        self,
        nodes: List[CapabilityNode],
        area_size: Tuple[float, float],
        grid_size: Tuple[int, int] = (10, 10)
    ) -> List[SpatialCell]:
        """
        生成空间网格（用于空间网格索引/分区建模）
        
        Args:
            nodes: 节点列表
            area_size: 区域大小
            grid_size: 网格大小 (rows, cols)
            
        Returns:
            空间网格单元列表
        """
        cells = []
        cell_width = area_size[0] / grid_size[0]
        cell_height = area_size[1] / grid_size[1]
        
        cell_id = 0
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                x_min = col * cell_width
                y_min = row * cell_height
                x_max = (col + 1) * cell_width
                y_max = (row + 1) * cell_height
                
                # 找出在此网格单元中的节点
                cell_nodes = []
                total_capability = 0.0
                for node in nodes:
                    x, y = node.position
                    if x_min <= x < x_max and y_min <= y < y_max:
                        cell_nodes.append(node.node_id)
                        total_capability += node.capability
                
                cell = SpatialCell(
                    cell_id=cell_id,
                    bounds=(x_min, y_min, x_max, y_max),
                    nodes=cell_nodes,
                    total_capability=total_capability
                )
                cells.append(cell)
                cell_id += 1
        
        return cells
    
    def generate_dynamic_scenario(
        self,
        num_nodes: int = 30,
        area_size: Tuple[float, float] = (1000, 1000),
        time_horizon: float = 200.0,
        grid_size: Tuple[int, int] = (10, 10)
    ) -> Dict:
        """
        生成完整的动态场景
        
        Args:
            num_nodes: 节点数量
            area_size: 区域大小
            time_horizon: 时间范围
            grid_size: 网格大小
            
        Returns:
            包含所有场景数据的字典
        """
        nodes, contacts, graph = self.generate_network_topology(
            num_nodes=num_nodes,
            area_size=area_size,
            time_horizon=time_horizon
        )
        spatial_grid = self.generate_spatial_grid(nodes, area_size, grid_size)
        
        scenario = {
            'nodes': nodes,
            'contacts': contacts,
            'graph': graph,
            'spatial_grid': spatial_grid,
            'num_nodes': num_nodes,
            'area_size': area_size,
            'time_horizon': time_horizon,
            'grid_size': grid_size
        }
        
        return scenario

