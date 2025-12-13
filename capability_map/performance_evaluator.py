"""
能力底图构建算法性能评估模块
"""

import numpy as np
from typing import List, Dict
from .scenario_generator import CapabilityNode


class CapabilityMapEvaluator:
    """能力底图构建性能评估器"""
    
    def __init__(self):
        """初始化性能评估器"""
        pass
    
    def evaluate_teg_cgr(
        self,
        results: List[Dict],
        queries: List[Dict]
    ) -> Dict[str, float]:
        """
        评估TEG/CGR建模性能
        
        Args:
            results: 查询结果列表
            queries: 查询列表
            
        Returns:
            性能指标字典
        """
        if len(results) != len(queries):
            raise ValueError("结果数量与查询数量不匹配")
        
        metrics = {}
        
        # 成功率
        feasible_count = sum(1 for r in results if r.get('feasible', False))
        metrics['success_rate'] = feasible_count / len(results) if len(results) > 0 else 0.0
        
        # 平均能力
        feasible_capabilities = [r['total_capability'] for r in results if r.get('feasible', False)]
        metrics['avg_capability'] = np.mean(feasible_capabilities) if feasible_capabilities else 0.0
        
        # 平均带宽
        feasible_bandwidths = [r['total_bandwidth'] for r in results if r.get('feasible', False)]
        metrics['avg_bandwidth'] = np.mean(feasible_bandwidths) if feasible_bandwidths else 0.0
        
        # 平均接触数
        feasible_contacts = [r['contact_count'] for r in results if r.get('feasible', False)]
        metrics['avg_contact_count'] = np.mean(feasible_contacts) if feasible_contacts else 0.0
        
        # 平均路径长度
        feasible_paths = [len(r['path']) for r in results if r.get('feasible', False) and r.get('path')]
        metrics['avg_path_length'] = np.mean(feasible_paths) if feasible_paths else 0.0
        
        return metrics
    
    def evaluate_spatial_hashing(
        self,
        capability_map: Dict[int, float],
        query_results: List[Dict]
    ) -> Dict[str, float]:
        """
        评估空间网格索引性能
        
        Args:
            capability_map: 能力底图
            query_results: 查询结果列表
            
        Returns:
            性能指标字典
        """
        metrics = {}
        
        # 能力底图统计
        capability_values = list(capability_map.values())
        metrics['total_cells'] = len(capability_map)
        metrics['avg_cell_capability'] = np.mean(capability_values) if capability_values else 0.0
        metrics['max_cell_capability'] = np.max(capability_values) if capability_values else 0.0
        metrics['min_cell_capability'] = np.min(capability_values) if capability_values else 0.0
        
        # 查询性能
        if query_results:
            query_capabilities = [r.get('total_capability', 0) for r in query_results]
            query_node_counts = [r.get('node_count', 0) for r in query_results]
            
            metrics['avg_query_capability'] = np.mean(query_capabilities) if query_capabilities else 0.0
            metrics['avg_query_node_count'] = np.mean(query_node_counts) if query_node_counts else 0.0
        
        return metrics
    
    def compare_algorithms(
        self,
        algorithm_results: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        对比多个算法的性能
        
        Args:
            algorithm_results: {算法名称: 性能指标字典}
            
        Returns:
            对比结果字典
        """
        return algorithm_results

