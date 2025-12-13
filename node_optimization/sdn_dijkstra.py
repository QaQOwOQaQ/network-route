"""
SDN架构下的Dijkstra节点优化排序算法
在SDN控制器管理下进行节点优化
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .scenario_generator import Node, Task, SDNController


class SDNDijkstraOptimizer:
    """SDN架构下的Dijkstra优化器"""
    
    def __init__(self):
        """初始化SDN Dijkstra优化器"""
        pass
    
    def _calculate_node_cost(
        self,
        node: Node,
        task: Task,
        controller: Optional[SDNController] = None
    ) -> float:
        """
        计算节点代价
        
        Args:
            node: 节点
            task: 任务
            controller: SDN控制器（如果节点在其覆盖范围内）
            
        Returns:
            节点代价
        """
        # 检查资源是否足够
        if (node.capacity * (1 - node.current_load) < task.required_capacity or
            node.memory < task.required_memory or
            node.bandwidth < task.required_bandwidth):
            return float('inf')
        
        # 计算代价：负载、距离、能量、优先级
        load_cost = node.current_load * 100
        energy_cost = (1 - node.energy_level) * 50
        
        # 如果节点在SDN控制器覆盖下，降低代价
        sdn_bonus = 0.0
        if controller and node.node_id in controller.coverage_nodes:
            sdn_bonus = -20.0  # SDN管理的节点有优势
        
        # 优先级越高，代价越低
        priority_cost = (6 - task.priority) * 10
        
        total_cost = load_cost + energy_cost + priority_cost - sdn_bonus
        
        return total_cost
    
    def optimize_ordering(
        self,
        nodes: List[Node],
        tasks: List[Task],
        controllers: List[SDNController]
    ) -> List[Tuple[int, int]]:
        """
        使用SDN架构下的Dijkstra算法优化节点排序
        
        Args:
            nodes: 节点列表
            tasks: 任务列表
            controllers: SDN控制器列表
            
        Returns:
            [(任务ID, 节点ID)] 排序结果
        """
        assignments = []
        node_dict = {node.node_id: node for node in nodes}
        
        for task in tasks:
            best_node_id = None
            best_cost = float('inf')
            
            # 为每个节点计算代价
            for node in nodes:
                # 找到管理该节点的控制器
                controller = None
                for ctrl in controllers:
                    if node.node_id in ctrl.coverage_nodes:
                        controller = ctrl
                        break
                
                cost = self._calculate_node_cost(node, task, controller)
                
                if cost < best_cost:
                    best_cost = cost
                    best_node_id = node.node_id
            
            if best_node_id is not None and best_cost != float('inf'):
                assignments.append((task.task_id, best_node_id))
        
        return assignments
    
    def optimize_with_global_view(
        self,
        nodes: List[Node],
        tasks: List[Task],
        controllers: List[SDNController]
    ) -> List[Tuple[int, int]]:
        """
        使用SDN全局视图优化（考虑负载均衡）
        
        Args:
            nodes: 节点列表
            tasks: 任务列表
            controllers: SDN控制器列表
            
        Returns:
            [(任务ID, 节点ID)] 排序结果
        """
        assignments = []
        node_loads = {node.node_id: node.current_load for node in nodes}
        
        # 按优先级排序任务
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            best_node_id = None
            best_score = -float('inf')
            
            for node in nodes:
                # 检查资源
                if (node.capacity * (1 - node.current_load) < task.required_capacity or
                    node.memory < task.required_memory or
                    node.bandwidth < task.required_bandwidth):
                    continue
                
                # 计算综合得分（考虑负载均衡）
                controller = None
                for ctrl in controllers:
                    if node.node_id in ctrl.coverage_nodes:
                        controller = ctrl
                        break
                
                # 得分：资源充足度、能量、SDN管理、负载均衡
                capacity_score = (node.capacity * (1 - node.current_load)) / task.required_capacity
                energy_score = node.energy_level
                sdn_score = 1.2 if controller else 1.0
                load_balance_score = 1.0 - node.current_load  # 负载越低越好
                
                total_score = capacity_score * energy_score * sdn_score * load_balance_score * task.priority
                
                if total_score > best_score:
                    best_score = total_score
                    best_node_id = node.node_id
            
            if best_node_id is not None:
                assignments.append((task.task_id, best_node_id))
                # 更新节点负载（简化）
                node = next(n for n in nodes if n.node_id == best_node_id)
                node.current_load += task.required_capacity / node.capacity
        
        return assignments

