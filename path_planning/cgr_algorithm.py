"""
接触图路由 (Contact Graph Routing, CGR) 算法实现
适用于延迟容忍网络 (DTN)
"""

import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import heapq
from .scenario_generator import Node, Link, Task


@dataclass
class PathResult:
    """路径规划结果"""
    path: List[int]  # 节点ID序列
    total_delay: float  # 总延迟
    total_cost: float  # 总代价
    feasible: bool  # 是否可行
    arrival_time: float  # 到达时间


class CGRAlgorithm:
    """接触图路由算法"""
    
    def __init__(self, nodes: List[Node], links: List[Link], graph: nx.Graph):
        """
        初始化CGR算法
        
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
    
    def _is_link_available(self, source: int, target: int, current_time: float) -> bool:
        """检查链路在指定时间是否可用"""
        link_key = (source, target)
        if link_key not in self.link_dict:
            return False
        
        link = self.link_dict[link_key]
        return link.start_time <= current_time <= link.end_time
    
    def _get_link_delay(self, source: int, target: int) -> float:
        """获取链路延迟"""
        link_key = (source, target)
        if link_key in self.link_dict:
            return self.link_dict[link_key].delay
        return float('inf')
    
    def find_path(
        self,
        task: Task,
        current_time: float = 0.0,
        max_hops: int = 10
    ) -> PathResult:
        """
        使用CGR算法查找路径
        
        Args:
            task: 任务
            current_time: 当前时间
            max_hops: 最大跳数
            
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
        
        # 使用改进的Dijkstra算法，考虑时间窗口
        # 优先级队列: (到达时间, 节点, 路径, 跳数)
        pq = [(current_time, source, [source], 0)]
        visited = set()
        
        best_path = None
        best_arrival_time = float('inf')
        
        # 添加最大迭代次数限制，防止无限循环
        max_iterations = 10000
        iteration_count = 0
        
        while pq and iteration_count < max_iterations:
            iteration_count += 1
            arrival_time, node, path, hops = heapq.heappop(pq)
            
            if node == destination:
                if arrival_time < best_arrival_time:
                    best_arrival_time = arrival_time
                    best_path = path
                # 如果找到路径且时间已经很好，可以提前退出
                if best_arrival_time <= current_time + task.deadline * 0.5:
                    break
                continue
            
            if hops >= max_hops:
                continue
            
            # 优化：使用更粗粒度的状态表示，减少状态空间
            time_bucket = int(arrival_time / 10)  # 将时间离散化为10ms的桶
            state = (node, time_bucket)
            if state in visited:
                continue
            visited.add(state)
            
            # 检查节点是否可用
            if node in self.nodes:
                node_obj = self.nodes[node]
                if not (node_obj.available_time <= arrival_time <= node_obj.unavailable_time):
                    # 节点不可用，等待到可用时间
                    arrival_time = max(arrival_time, node_obj.available_time)
            
            # 遍历邻居节点
            if node in self.graph:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in path:  # 避免循环
                        continue
                    
                    # 检查链路是否可用
                    if not self._is_link_available(node, neighbor, arrival_time):
                        continue
                    
                    link_delay = self._get_link_delay(node, neighbor)
                    new_arrival_time = arrival_time + link_delay
                    
                    # 检查是否超过截止时间
                    if new_arrival_time > task.deadline:
                        continue
                    
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_arrival_time, neighbor, new_path, hops + 1))
        
        if best_path is None:
            return PathResult(
                path=[],
                total_delay=float('inf'),
                total_cost=float('inf'),
                feasible=False,
                arrival_time=float('inf')
            )
        
        # 计算总延迟和代价
        total_delay = best_arrival_time - current_time
        total_cost = self._calculate_path_cost(best_path, current_time)
        
        return PathResult(
            path=best_path,
            total_delay=total_delay,
            total_cost=total_cost,
            feasible=True,
            arrival_time=best_arrival_time
        )
    
    def _calculate_path_cost(self, path: List[int], start_time: float) -> float:
        """计算路径代价"""
        if len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        current_time = start_time
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            link_key = (source, target)
            if link_key in self.link_dict:
                link = self.link_dict[link_key]
                # 代价 = 延迟 + (1 - 可靠性) * 惩罚
                cost = link.delay + (1 - link.reliability) * 10
                total_cost += cost
                current_time += link.delay
        
        return total_cost

