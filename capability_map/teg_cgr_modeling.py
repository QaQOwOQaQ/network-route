"""
时间演进图 (TEG) / 接触图路由 (CGR) 建模算法
构建时间扩展的能力底图
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from .scenario_generator import CapabilityNode, Contact


class TEGCGRModeling:
    """时间演进图/接触图路由建模"""
    
    def __init__(self, nodes: List[CapabilityNode], contacts: List[Contact], graph: nx.Graph):
        """
        初始化TEG/CGR建模
        
        Args:
            nodes: 节点列表
            contacts: 接触列表
            graph: 网络图
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.contacts = contacts
        self.graph = graph
        self.teg_graph = None  # 时间演进图
        self.cgr_graph = None  # 接触图
    
    def build_time_expanded_graph(
        self,
        time_steps: List[float],
        time_window: float = 10.0
    ) -> nx.DiGraph:
        """
        构建时间演进图 (Time-Expanded Graph, TEG)
        
        Args:
            time_steps: 时间步列表
            time_window: 时间窗口大小
            
        Returns:
            时间演进有向图
        """
        teg = nx.DiGraph()
        
        # 为每个时间步创建节点副本
        for t in time_steps:
            for node_id, node in self.nodes.items():
                # 节点在时间t的副本
                teg_node = (node_id, t)
                teg.add_node(teg_node, 
                           capability=node.capability,
                           time=t)
        
        # 添加时间内的边（节点在同一时间步内的连接）
        for contact in self.contacts:
            # 找出接触活跃的时间步
            for t in time_steps:
                if contact.start_time <= t <= contact.end_time:
                    source_node = (contact.source, t)
                    target_node = (contact.target, t)
                    
                    if source_node in teg and target_node in teg:
                        teg.add_edge(source_node, target_node,
                                   bandwidth=contact.bandwidth,
                                   distance=contact.distance,
                                   contact_id=id(contact))
        
        # 添加时间演进边（节点在不同时间步之间的连接）
        sorted_time_steps = sorted(time_steps)
        for i in range(len(sorted_time_steps) - 1):
            t1 = sorted_time_steps[i]
            t2 = sorted_time_steps[i + 1]
            
            # 同一节点在不同时间步之间的连接
            for node_id in self.nodes.keys():
                node1 = (node_id, t1)
                node2 = (node_id, t2)
                
                if node1 in teg and node2 in teg:
                    # 检查节点在时间窗口内是否可用
                    node = self.nodes[node_id]
                    if node.available_time <= t2 <= node.unavailable_time:
                        teg.add_edge(node1, node2,
                                   bandwidth=float('inf'),  # 节点内部传输
                                   distance=0.0,
                                   time_evolution=True)
        
        self.teg_graph = teg
        return teg
    
    def build_contact_graph(self) -> nx.Graph:
        """
        构建接触图 (Contact Graph)
        
        Returns:
            接触图
        """
        cgr = nx.Graph()
        
        # 添加节点
        for node_id, node in self.nodes.items():
            cgr.add_node(node_id,
                        position=node.position,
                        capability=node.capability,
                        contact_windows=node.contact_windows)
        
        # 添加接触边
        for contact in self.contacts:
            if contact.source in cgr and contact.target in cgr:
                cgr.add_edge(contact.source, contact.target,
                           start_time=contact.start_time,
                           end_time=contact.end_time,
                           bandwidth=contact.bandwidth,
                           distance=contact.distance,
                           duration=contact.end_time - contact.start_time)
        
        self.cgr_graph = cgr
        return cgr
    
    def query_capability(
        self,
        source: int,
        destination: int,
        start_time: float,
        end_time: float
    ) -> Dict:
        """
        查询从源节点到目标节点的能力
        
        Args:
            source: 源节点
            destination: 目标节点
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            能力信息字典
        """
        if self.cgr_graph is None:
            self.build_contact_graph()
        
        # 使用CGR查找路径
        if source not in self.cgr_graph or destination not in self.cgr_graph:
            return {
                'feasible': False,
                'path': [],
                'total_capability': 0.0,
                'total_bandwidth': 0.0,
                'contact_count': 0
            }
        
        # 简化的路径查找（考虑时间窗口）
        try:
            # 使用最短路径，但考虑接触时间窗口
            path = nx.shortest_path(self.cgr_graph, source, destination)
            
            # 计算路径能力
            total_capability = 0.0
            min_bandwidth = float('inf')
            contact_count = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.cgr_graph.has_edge(u, v):
                    edge_data = self.cgr_graph[u][v]
                    # 检查时间窗口
                    if edge_data['start_time'] <= start_time <= edge_data['end_time']:
                        total_capability += self.nodes[u].capability
                        min_bandwidth = min(min_bandwidth, edge_data['bandwidth'])
                        contact_count += 1
            
            if destination in self.nodes:
                total_capability += self.nodes[destination].capability
            
            return {
                'feasible': len(path) > 1,
                'path': path,
                'total_capability': total_capability,
                'total_bandwidth': min_bandwidth if min_bandwidth != float('inf') else 0.0,
                'contact_count': contact_count
            }
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {
                'feasible': False,
                'path': [],
                'total_capability': 0.0,
                'total_bandwidth': 0.0,
                'contact_count': 0
            }
    
    def get_capability_map(self, time: float) -> Dict[int, float]:
        """
        获取指定时间的能力底图
        
        Args:
            time: 时间点
            
        Returns:
            {节点ID: 能力值}
        """
        capability_map = {}
        for node_id, node in self.nodes.items():
            if node.available_time <= time <= node.unavailable_time:
                capability_map[node_id] = node.capability
            else:
                capability_map[node_id] = 0.0
        
        return capability_map

