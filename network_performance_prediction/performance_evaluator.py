"""
网络性能预测算法性能评估模块
"""

import numpy as np
from typing import List, Dict
from .scenario_generator import NetworkState


class PerformancePredictionEvaluator:
    """性能预测评估器"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate_algorithm(
        self,
        predictions: List[Dict[str, float]],
        actual_states: List[NetworkState]
    ) -> Dict[str, float]:
        """
        评估算法整体性能
        
        Args:
            predictions: 预测结果列表
            actual_states: 实际网络状态列表
            
        Returns:
            性能指标字典
        """
        if len(predictions) != len(actual_states):
            raise ValueError("预测结果和实际状态数量不匹配")
        
        # 提取延迟数据
        delays_pred = []
        delays_actual = []
        
        for pred, actual in zip(predictions, actual_states):
            if 'delay' in pred and pred['delay'] != float('inf') and not np.isnan(pred['delay']):
                delays_pred.append(pred['delay'])
                delays_actual.append(actual.delay)
        
        if len(delays_pred) == 0:
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'r2_score': -float('inf')
            }
        
        delays_pred = np.array(delays_pred)
        delays_actual = np.array(delays_actual)
        
        # 计算误差指标
        mae = np.mean(np.abs(delays_pred - delays_actual))
        mse = np.mean((delays_pred - delays_actual) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((delays_actual - delays_pred) / delays_actual)) * 100
            mape = mape if not np.isnan(mape) else float('inf')
        
        # R² 分数
        ss_res = np.sum((delays_actual - delays_pred) ** 2)
        ss_tot = np.sum((delays_actual - np.mean(delays_actual)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else -float('inf')
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2_score
        }
    
    def compare_algorithms(
        self,
        algorithm_results: Dict[str, List[Dict[str, float]]],
        actual_states: List[NetworkState]
    ) -> Dict[str, Dict]:
        """
        对比多个算法的性能
        
        Args:
            algorithm_results: {算法名称: 预测结果列表}
            actual_states: 实际网络状态列表
            
        Returns:
            对比结果字典
        """
        comparison = {}
        
        for algo_name, predictions in algorithm_results.items():
            comparison[algo_name] = self.evaluate_algorithm(predictions, actual_states)
        
        return comparison

