"""
Dijkstra和A*路径规划算法实现
"""

import networkx as nx
from typing import List, Dict, Tuple, Optional
import heapq
import numpy as np
from .scenario_generator import Node, Link, Task
from .cgr_algorithm import PathResult


class DijkstraAlgorithm:
    """Dijkstra最短路径算法"""
    
    def __init__(self, nodes: List[Node], links: List[Link], graph: nx.Graph):
        """
        初始化Dijkstra算法
        
        Args:
            nodes: 节点列表
            links: 链路列表
            graph: 网络图
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.links = links
        self.graph = graph
        self.link_dict = self._build_link_dict()
    
    def _build_link_dict(self) -> Dict[Tuple[int, int], Link]:
        """构建链路字典"""
        link_dict = {}
        for link in self.links:
            link_dict[(link.source, link.target)] = link
            link_dict[(link.target, link.source)] = link
        return link_dict
    
    def _get_edge_weight(self, source: int, target: int, weight_type: str = 'delay') -> float:
        """获取边权重"""
        link_key = (source, target)
        if link_key not in self.link_dict:
            return float('inf')
        
        link = self.link_dict[link_key]
        if weight_type == 'delay':
            return link.delay
        elif weight_type == 'cost':
            # 综合代价：延迟 + 可靠性惩罚
            return link.delay + (1 - link.reliability) * 10
        else:
            return link.delay
    
    def find_path(
        self,
        task: Task,
        current_time: float = 0.0,
        weight_type: str = 'delay'
    ) -> PathResult:
        """
        使用Dijkstra算法查找最短路径
        
        Args:
            task: 任务
            current_time: 当前时间
            weight_type: 权重类型 ('delay' 或 'cost')
            
        Returns:
            路径结果
        """
        source = task.source
        destination = task.destination
        
        if source == destination:
            return PathResult(
                path=[source],
                total_delay=0.0,
                total_cost=0.0,
                feasible=True,
                arrival_time=current_time
            )
        
        # Dijkstra算法
        dist = {node: float('inf') for node in self.graph.nodes()}
        prev = {node: None for node in self.graph.nodes()}
        dist[source] = 0.0
        
        pq = [(0.0, source)]
        visited = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            if current_node == destination:
                break
            
            # 遍历邻居
            for neighbor in self.graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                weight = self._get_edge_weight(current_node, neighbor, weight_type)
                if weight == float('inf'):
                    continue
                
                alt = current_dist + weight
                if alt < dist[neighbor]:
                    dist[neighbor] = alt
                    prev[neighbor] = current_node
                    heapq.heappush(pq, (alt, neighbor))
        
        # 重构路径
        if dist[destination] == float('inf'):
            return PathResult(
                path=[],
                total_delay=float('inf'),
                total_cost=float('inf'),
                feasible=False,
                arrival_time=float('inf')
            )
        
        path = []
        node = destination
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        
        # 计算总延迟
        total_delay = 0.0
        for i in range(len(path) - 1):
            total_delay += self._get_edge_weight(path[i], path[i + 1], 'delay')
        
        total_cost = dist[destination]
        arrival_time = current_time + total_delay
        
        # 检查是否满足截止时间
        feasible = arrival_time <= task.deadline
        
        return PathResult(
            path=path,
            total_delay=total_delay,
            total_cost=total_cost,
            feasible=feasible,
            arrival_time=arrival_time
        )


class AStarAlgorithm:
    """A*路径规划算法"""
    
    def __init__(self, nodes: List[Node], links: List[Link], graph: nx.Graph):
        """
        初始化A*算法
        
        Args:
            nodes: 节点列表
            links: 链路列表
            graph: 网络图
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.links = links
        self.graph = graph
        self.link_dict = self._build_link_dict()
    
    def _build_link_dict(self) -> Dict[Tuple[int, int], Link]:
        """构建链路字典"""
        link_dict = {}
        for link in self.links:
            link_dict[(link.source, link.target)] = link
            link_dict[(link.target, link.source)] = link
        return link_dict
    
    def _heuristic(self, node1: int, node2: int) -> float:
        """启发式函数：欧几里得距离 + 可靠性因子"""
        if node1 not in self.nodes or node2 not in self.nodes:
            return 0.0
        
        pos1 = self.nodes[node1].position
        pos2 = self.nodes[node2].position
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # 添加可靠性因子，使A*倾向于选择更可靠的路径
        # 检查直接链路
        link_key = (node1, node2)
        if link_key in self.link_dict:
            reliability = self.link_dict[link_key].reliability
            # 可靠性越低，启发式值越高（惩罚）
            reliability_penalty = (1 - reliability) * 5
            return distance + reliability_penalty
        
        return distance
    
    def _get_edge_weight(self, source: int, target: int) -> float:
        """获取边权重"""
        link_key = (source, target)
        if link_key not in self.link_dict:
            return float('inf')
        return self.link_dict[link_key].delay
    
    def find_path(
        self,
        task: Task,
        current_time: float = 0.0
    ) -> PathResult:
        """
        使用A*算法查找路径
        
        Args:
            task: 任务
            current_time: 当前时间
            
        Returns:
            路径结果
        """
        source = task.source
        destination = task.destination
        
        if source == destination:
            return PathResult(
                path=[source],
                total_delay=0.0,
                total_cost=0.0,
                feasible=True,
                arrival_time=current_time
            )
        
        # A*算法
        open_set = [(0.0, source)]  # (f_score, node)
        came_from = {}
        g_score = {node: float('inf') for node in self.graph.nodes()}
        g_score[source] = 0.0
        f_score = {node: float('inf') for node in self.graph.nodes()}
        f_score[source] = self._heuristic(source, destination)
        
        visited = set()
        
        while open_set:
            current_f, current_node = heapq.heappop(open_set)
            
            if current_node == destination:
                # 重构路径
                path = []
                node = destination
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(source)
                path.reverse()
                
                # 计算总延迟
                total_delay = g_score[destination]
                arrival_time = current_time + total_delay
                feasible = arrival_time <= task.deadline
                
                return PathResult(
                    path=path,
                    total_delay=total_delay,
                    total_cost=total_delay,
                    feasible=feasible,
                    arrival_time=arrival_time
                )
            
            visited.add(current_node)
            
            for neighbor in self.graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                weight = self._get_edge_weight(current_node, neighbor)
                if weight == float('inf'):
                    continue
                
                tentative_g = g_score[current_node] + weight
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, destination)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 未找到路径
        return PathResult(
            path=[],
            total_delay=float('inf'),
            total_cost=float('inf'),
            feasible=False,
            arrival_time=float('inf')
        )

