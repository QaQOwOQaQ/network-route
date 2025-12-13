"""
节点优化排序算法性能评估模块
"""

import numpy as np
from typing import List, Dict, Tuple
from .scenario_generator import Node, Task


class NodeOptimizationEvaluator:
    """节点优化排序评估器"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate_assignment(
        self,
        assignments: List[Tuple[int, int]],
        nodes: List[Node],
        tasks: List[Task]
    ) -> Dict[str, float]:
        """
        评估节点分配结果
        
        Args:
            assignments: [(任务ID, 节点ID)] 分配结果
            nodes: 节点列表
            tasks: 任务列表
            
        Returns:
            性能指标字典
        """
        node_dict = {node.node_id: node for node in nodes}
        task_dict = {task.task_id: task for task in tasks}
        
        # 统计指标
        total_tasks = len(tasks)
        assigned_tasks = len(assignments)
        success_rate = assigned_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # 计算负载均衡度
        node_loads = {node.node_id: node.current_load for node in nodes}
        node_task_counts = {node.node_id: 0 for node in nodes}
        
        for task_id, node_id in assignments:
            if node_id in node_task_counts:
                node_task_counts[node_id] += 1
                task = task_dict.get(task_id)
                if task:
                    node = node_dict[node_id]
                    node_loads[node_id] += task.required_capacity / node.capacity
        
        # 负载均衡度（标准差越小越好）
        if len(node_loads) > 0:
            load_values = list(node_loads.values())
            load_balance = 1.0 / (1.0 + np.std(load_values))  # 归一化
        else:
            load_balance = 0.0
        
        # 平均资源利用率
        avg_utilization = np.mean(list(node_loads.values())) if node_loads else 0.0
        
        # 优先级满足率（高优先级任务是否优先分配）
        priority_satisfaction = 0.0
        if assigned_tasks > 0:
            assigned_priorities = []
            for task_id, _ in assignments:
                task = task_dict.get(task_id)
                if task:
                    assigned_priorities.append(task.priority)
            
            if assigned_priorities:
                avg_assigned_priority = np.mean(assigned_priorities)
                max_priority = max([t.priority for t in tasks])
                priority_satisfaction = avg_assigned_priority / max_priority
        
        # 能量效率（使用能量水平高的节点）
        energy_efficiency = 0.0
        if assigned_tasks > 0:
            used_energies = []
            for task_id, node_id in assignments:
                node = node_dict.get(node_id)
                if node:
                    used_energies.append(node.energy_level)
            
            if used_energies:
                energy_efficiency = np.mean(used_energies)
        
        return {
            'success_rate': success_rate,
            'load_balance': load_balance,
            'avg_utilization': avg_utilization,
            'priority_satisfaction': priority_satisfaction,
            'energy_efficiency': energy_efficiency,
            'assigned_count': assigned_tasks,
            'total_count': total_tasks
        }
    
    def compare_algorithms(
        self,
        algorithm_results: Dict[str, List[Tuple[int, int]]],
        nodes: List[Node],
        tasks: List[Task]
    ) -> Dict[str, Dict]:
        """
        对比多个算法的性能
        
        Args:
            algorithm_results: {算法名称: 分配结果列表}
            nodes: 节点列表
            tasks: 任务列表
            
        Returns:
            对比结果字典
        """
        comparison = {}
        
        for algo_name, assignments in algorithm_results.items():
            comparison[algo_name] = self.evaluate_assignment(assignments, nodes, tasks)
        
        return comparison

