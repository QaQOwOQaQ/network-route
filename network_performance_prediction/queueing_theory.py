"""
排队论网络性能预测算法
基于M/M/1和M/M/c排队模型
"""

import numpy as np
from typing import List, Dict
from .scenario_generator import NetworkState, TrafficPattern


class QueueingTheoryPredictor:
    """排队论性能预测器"""
    
    def __init__(self, max_delay: float = 2000.0):
        """
        初始化排队论预测器
        
        Args:
            max_delay: 最大延迟上限（ms），防止异常大的预测值
        """
        self.max_delay = max_delay
    
    def predict_mm1(
        self,
        arrival_rate: float,
        service_rate: float
    ) -> Dict[str, float]:
        """
        使用M/M/1模型预测性能（改进版，考虑丢包和延迟上限）
        
        Args:
            arrival_rate: 到达率 (packets/s)
            service_rate: 服务率 (packets/s)
            
        Returns:
            性能指标字典
        """
        # 确保输入值有效
        arrival_rate = max(0.0, arrival_rate)
        service_rate = max(0.001, service_rate)  # 避免除以0
        
        if arrival_rate >= service_rate:
            # 系统接近饱和，考虑丢包和流量控制
            # 实际网络中会有丢包，有效到达率会降低
            effective_arrival_rate = service_rate * 0.95  # 假设5%丢包
            utilization = 0.95
            packet_loss = 0.05 + (arrival_rate - service_rate) / arrival_rate * 0.1
            packet_loss = min(packet_loss, 0.5)  # 最大丢包率50%
        else:
            effective_arrival_rate = arrival_rate
            utilization = arrival_rate / service_rate
            # 丢包率：当利用率高时开始丢包
            packet_loss = 0.0 if utilization < 0.85 else (utilization - 0.85) * 0.2
            packet_loss = min(packet_loss, 0.3)  # 最大丢包率30%
        
        # 考虑丢包后的有效利用率
        effective_utilization = utilization * (1 - packet_loss)
        
        # 限制利用率上限，避免极端值
        effective_utilization = min(effective_utilization, 0.98)
        
        # 平均队列长度（考虑丢包）
        if effective_utilization < 1.0:
            queue_length = effective_utilization / (1 - effective_utilization)
        else:
            queue_length = 50.0  # 饱和状态下的合理队列长度
        
        # 平均延迟（Little's Law），使用有效到达率
        if effective_arrival_rate > 0:
            delay = queue_length / effective_arrival_rate * 1000  # 转换为ms
        else:
            delay = 0.0
        
        # 应用延迟上限，防止异常大的值
        delay = min(delay, self.max_delay)
        
        # 吞吐量（考虑丢包）
        throughput = effective_arrival_rate * (1 - packet_loss) * 8 / 1e6  # 转换为Mbps
        
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
        使用M/M/c模型预测性能（多服务器，改进版）
        
        Args:
            arrival_rate: 到达率
            service_rate: 单个服务器服务率
            num_servers: 服务器数量
            
        Returns:
            性能指标字典
        """
        # 确保输入值有效
        arrival_rate = max(0.0, arrival_rate)
        service_rate = max(0.001, service_rate)
        num_servers = max(1, num_servers)
        
        total_service_rate = service_rate * num_servers
        
        if arrival_rate >= total_service_rate:
            # 系统接近饱和，考虑丢包
            effective_arrival_rate = total_service_rate * 0.95
            utilization = 0.95
            packet_loss = 0.05 + (arrival_rate - total_service_rate) / arrival_rate * 0.1
            packet_loss = min(packet_loss, 0.5)
        else:
            effective_arrival_rate = arrival_rate
            utilization = arrival_rate / total_service_rate
            packet_loss = 0.0 if utilization < 0.85 else (utilization - 0.85) * 0.2
            packet_loss = min(packet_loss, 0.3)
        
        effective_utilization = utilization * (1 - packet_loss)
        effective_utilization = min(effective_utilization, 0.98)
        
        rho = effective_arrival_rate / service_rate  # 流量强度
        
        # 计算Erlang C公式（简化版）
        if rho < num_servers * 0.98:
            # 平均队列长度（近似）
            queue_length = (rho ** num_servers) / (num_servers * (1 - rho / num_servers))
            queue_length = min(queue_length, 100.0)  # 限制队列长度
        else:
            queue_length = 50.0  # 饱和状态
        
        # 平均延迟
        if effective_arrival_rate > 0:
            delay = queue_length / effective_arrival_rate * 1000
        else:
            delay = 0.0
        
        # 应用延迟上限
        delay = min(delay, self.max_delay)
        
        # 吞吐量
        throughput = effective_arrival_rate * (1 - packet_loss) * 8 / 1e6
        
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

