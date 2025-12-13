"""
性能指标评估模块
评估路径规划算法的性能指标
"""

import numpy as np
from typing import List, Dict
from .cgr_algorithm import PathResult
from .scenario_generator import Task


class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self):
        """初始化性能评估器"""
        pass
    
    def evaluate_single_result(
        self,
        result: PathResult,
        task: Task
    ) -> Dict[str, float]:
        """
        评估单个路径规划结果
        
        Args:
            result: 路径规划结果
            task: 任务
            
        Returns:
            性能指标字典
        """
        metrics = {}
        
        # 1. 路径可行性
        metrics['feasible'] = 1.0 if result.feasible else 0.0
        
        # 2. 总延迟
        metrics['total_delay'] = result.total_delay if result.feasible else float('inf')
        
        # 3. 总代价
        metrics['total_cost'] = result.total_cost if result.feasible else float('inf')
        
        # 4. 路径长度（跳数）
        metrics['path_length'] = len(result.path) - 1 if result.feasible else float('inf')
        
        # 5. 是否满足截止时间
        metrics['meets_deadline'] = 1.0 if result.feasible and result.arrival_time <= task.deadline else 0.0
        
        # 6. 时间余量（deadline - arrival_time）
        if result.feasible:
            metrics['time_margin'] = task.deadline - result.arrival_time
        else:
            metrics['time_margin'] = -float('inf')
        
        # 7. 平均每跳延迟
        if result.feasible and metrics['path_length'] > 0:
            metrics['avg_delay_per_hop'] = result.total_delay / metrics['path_length']
        else:
            metrics['avg_delay_per_hop'] = float('inf')
        
        return metrics
    
    def evaluate_algorithm(
        self,
        results: List[PathResult],
        tasks: List[Task]
    ) -> Dict[str, float]:
        """
        评估算法整体性能
        
        Args:
            results: 路径规划结果列表
            tasks: 任务列表
            
        Returns:
            整体性能指标字典
        """
        if len(results) != len(tasks):
            raise ValueError("结果数量与任务数量不匹配")
        
        all_metrics = []
        for result, task in zip(results, tasks):
            metrics = self.evaluate_single_result(result, task)
            all_metrics.append(metrics)
        
        # 聚合指标
        aggregated = {}
        
        # 成功率
        feasible_count = sum(1 for m in all_metrics if m['feasible'] > 0)
        aggregated['success_rate'] = feasible_count / len(results) if len(results) > 0 else 0.0
        
        # 截止时间满足率
        deadline_met_count = sum(1 for m in all_metrics if m['meets_deadline'] > 0)
        aggregated['deadline_satisfaction_rate'] = deadline_met_count / len(results) if len(results) > 0 else 0.0
        
        # 平均延迟（仅考虑可行路径）
        feasible_delays = [m['total_delay'] for m in all_metrics if m['feasible'] > 0]
        aggregated['avg_delay'] = np.mean(feasible_delays) if feasible_delays else float('inf')
        aggregated['median_delay'] = np.median(feasible_delays) if feasible_delays else float('inf')
        aggregated['std_delay'] = np.std(feasible_delays) if feasible_delays else 0.0
        
        # 平均代价
        feasible_costs = [m['total_cost'] for m in all_metrics if m['feasible'] > 0]
        aggregated['avg_cost'] = np.mean(feasible_costs) if feasible_costs else float('inf')
        
        # 平均路径长度
        feasible_lengths = [m['path_length'] for m in all_metrics if m['feasible'] > 0]
        aggregated['avg_path_length'] = np.mean(feasible_lengths) if feasible_lengths else float('inf')
        
        # 平均时间余量
        feasible_margins = [m['time_margin'] for m in all_metrics if m['feasible'] > 0]
        aggregated['avg_time_margin'] = np.mean(feasible_margins) if feasible_margins else float('inf')
        
        # 平均每跳延迟
        feasible_hop_delays = [m['avg_delay_per_hop'] for m in all_metrics if m['feasible'] > 0]
        aggregated['avg_delay_per_hop'] = np.mean(feasible_hop_delays) if feasible_hop_delays else float('inf')
        
        return aggregated
    
    def compare_algorithms(
        self,
        algorithm_results: Dict[str, List[PathResult]],
        tasks: List[Task]
    ) -> Dict[str, Dict]:
        """
        对比多个算法的性能
        
        Args:
            algorithm_results: {算法名称: 结果列表}
            tasks: 任务列表
            
        Returns:
            对比结果字典
        """
        comparison = {}
        
        for algo_name, results in algorithm_results.items():
            comparison[algo_name] = self.evaluate_algorithm(results, tasks)
        
        return comparison

