"""
排队论网络性能预测算法
基于M/M/1和M/M/c排队模型
"""

import numpy as np
from typing import List, Dict
from .scenario_generator import NetworkState, TrafficPattern


class QueueingTheoryPredictor:
    """排队论性能预测器"""
    
    def __init__(self):
        """初始化排队论预测器"""
        pass
    
    def predict_mm1(
        self,
        arrival_rate: float,
        service_rate: float
    ) -> Dict[str, float]:
        """
        使用M/M/1模型预测性能
        
        Args:
            arrival_rate: 到达率 (packets/s)
            service_rate: 服务率 (packets/s)
            
        Returns:
            性能指标字典
        """
        if arrival_rate >= service_rate:
            # 系统不稳定
            return {
                'utilization': 1.0,
                'queue_length': float('inf'),
                'delay': float('inf'),
                'throughput': service_rate,
                'packet_loss': 1.0
            }
        
        utilization = arrival_rate / service_rate
        
        # 平均队列长度
        queue_length = utilization / (1 - utilization)
        
        # 平均延迟（Little's Law）
        delay = queue_length / arrival_rate * 1000  # 转换为ms
        
        # 吞吐量
        throughput = arrival_rate * 8 / 1e6  # 转换为Mbps
        
        # 丢包率（简化模型）
        packet_loss = 0.0 if utilization < 0.9 else (utilization - 0.9) * 0.1
        
        return {
            'utilization': utilization,
            'queue_length': queue_length,
            'delay': delay,
            'throughput': throughput,
            'packet_loss': packet_loss
        }
    
    def predict_mmc(
        self,
        arrival_rate: float,
        service_rate: float,
        num_servers: int = 2
    ) -> Dict[str, float]:
        """
        使用M/M/c模型预测性能（多服务器）
        
        Args:
            arrival_rate: 到达率
            service_rate: 单个服务器服务率
            num_servers: 服务器数量
            
        Returns:
            性能指标字典
        """
        total_service_rate = service_rate * num_servers
        
        if arrival_rate >= total_service_rate:
            return {
                'utilization': 1.0,
                'queue_length': float('inf'),
                'delay': float('inf'),
                'throughput': total_service_rate,
                'packet_loss': 1.0
            }
        
        utilization = arrival_rate / total_service_rate
        rho = arrival_rate / service_rate  # 流量强度
        
        # 计算Erlang C公式（简化版）
        if rho < num_servers:
            # 平均队列长度（近似）
            queue_length = (rho ** num_servers) / (num_servers * (1 - rho / num_servers))
        else:
            queue_length = float('inf')
        
        # 平均延迟
        if arrival_rate > 0:
            delay = queue_length / arrival_rate * 1000
        else:
            delay = 0.0
        
        # 吞吐量
        throughput = arrival_rate * 8 / 1e6
        
        # 丢包率
        packet_loss = 0.0 if utilization < 0.9 else (utilization - 0.9) * 0.1
        
        return {
            'utilization': utilization,
            'queue_length': queue_length,
            'delay': delay,
            'throughput': throughput,
            'packet_loss': packet_loss
        }
    
    def predict_batch(
        self,
        states: List[NetworkState],
        model_type: str = 'mm1'
    ) -> List[Dict[str, float]]:
        """
        批量预测
        
        Args:
            states: 网络状态列表
            model_type: 模型类型 ('mm1' 或 'mmc')
            
        Returns:
            预测结果列表
        """
        predictions = []
        
        for state in states:
            if model_type == 'mm1':
                pred = self.predict_mm1(state.arrival_rate, state.service_rate)
            else:
                pred = self.predict_mmc(state.arrival_rate, state.service_rate, num_servers=2)
            
            predictions.append(pred)
        
        return predictions
    
    def evaluate_prediction(
        self,
        predictions: List[Dict[str, float]],
        actual_states: List[NetworkState]
    ) -> Dict[str, float]:
        """
        评估预测准确性
        
        Args:
            predictions: 预测结果列表
            actual_states: 实际网络状态列表
            
        Returns:
            评估指标字典
        """
        if len(predictions) != len(actual_states):
            raise ValueError("预测结果和实际状态数量不匹配")
        
        delays_pred = [p['delay'] for p in predictions if p['delay'] != float('inf')]
        delays_actual = [s.delay for s in actual_states]
        
        # 对齐长度
        min_len = min(len(delays_pred), len(delays_actual))
        delays_pred = delays_pred[:min_len]
        delays_actual = delays_actual[:min_len]
        
        if len(delays_pred) == 0:
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf')
            }
        
        delays_pred = np.array(delays_pred)
        delays_actual = np.array(delays_actual)
        
        # 计算误差指标
        mae = np.mean(np.abs(delays_pred - delays_actual))
        mse = np.mean((delays_pred - delays_actual) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE（平均绝对百分比误差）
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((delays_actual - delays_pred) / delays_actual)) * 100
            mape = mape if not np.isnan(mape) else float('inf')
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }

