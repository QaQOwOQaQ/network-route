"""
网络性能预测场景生成模块
生成网络流量、负载、性能数据等
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random


@dataclass
class NetworkState:
    """网络状态"""
    timestamp: float
    node_id: int
    queue_length: float  # 队列长度
    arrival_rate: float  # 到达率 (packets/s)
    service_rate: float  # 服务率 (packets/s)
    utilization: float  # 利用率
    packet_loss: float  # 丢包率
    delay: float  # 延迟 (ms)
    throughput: float  # 吞吐量 (Mbps)


@dataclass
class TrafficPattern:
    """流量模式"""
    pattern_id: int
    source_node: int
    dest_node: int
    data_rate: float  # 数据速率 (Mbps)
    packet_size: float  # 数据包大小 (bytes)
    duration: float  # 持续时间 (s)
    start_time: float  # 开始时间


class PerformanceScenarioGenerator:
    """网络性能预测场景生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化场景生成器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_traffic_patterns(
        self,
        num_nodes: int = 20,
        num_patterns: int = 100,
        time_horizon: float = 1000.0,
        min_data_rate: float = 1.0,
        max_data_rate: float = 100.0
    ) -> List[TrafficPattern]:
        """
        生成流量模式
        
        Args:
            num_nodes: 节点数量
            num_patterns: 流量模式数量
            time_horizon: 时间范围
            min_data_rate: 最小数据速率
            max_data_rate: 最大数据速率
            
        Returns:
            流量模式列表
        """
        patterns = []
        node_ids = list(range(num_nodes))
        
        for i in range(num_patterns):
            source = np.random.choice(node_ids)
            dest = np.random.choice([n for n in node_ids if n != source])
            data_rate = np.random.uniform(min_data_rate, max_data_rate)
            packet_size = np.random.uniform(64, 1500)  # 64-1500 bytes
            duration = np.random.uniform(10, 100)
            start_time = np.random.uniform(0, time_horizon - duration)
            
            pattern = TrafficPattern(
                pattern_id=i,
                source_node=source,
                dest_node=dest,
                data_rate=data_rate,
                packet_size=packet_size,
                duration=duration,
                start_time=start_time
            )
            patterns.append(pattern)
        
        return patterns
    
    def generate_network_states(
        self,
        num_nodes: int = 20,
        time_steps: int = 100,
        time_interval: float = 1.0,
        base_service_rate: float = 1000.0
    ) -> List[NetworkState]:
        """
        生成网络状态序列
        
        Args:
            num_nodes: 节点数量
            time_steps: 时间步数
            time_interval: 时间间隔
            base_service_rate: 基础服务率
            
        Returns:
            网络状态列表
        """
        states = []
        
        for t in range(time_steps):
            timestamp = t * time_interval
            for node_id in range(num_nodes):
                # 生成到达率（随时间变化，但限制在合理范围内）
                # 使用更保守的范围，避免极端利用率
                arrival_rate = base_service_rate * (0.2 + 0.6 * np.sin(t / 10.0) * np.random.uniform(0.8, 1.0))
                arrival_rate = max(0.0, arrival_rate)  # 确保非负
                
                service_rate = base_service_rate * np.random.uniform(0.95, 1.05)
                service_rate = max(0.001, service_rate)  # 确保为正
                
                # 计算利用率，限制在合理范围内（最大0.95，避免接近1）
                utilization = min(arrival_rate / service_rate, 0.95)
                
                # 计算队列长度（M/M/1模型）
                if utilization < 1.0:
                    queue_length = utilization / (1 - utilization)
                else:
                    queue_length = 100.0  # 饱和状态
                
                # 计算丢包率（当利用率高时）
                packet_loss = max(0, (utilization - 0.8) * 0.5) if utilization > 0.8 else 0.0
                
                # 计算"实际"延迟（模拟真实网络行为，而不是直接使用理论公式）
                # 添加随机波动、网络拥塞、路由变化等因素的影响
                if arrival_rate > 0:
                    # 基础延迟（Little's Law）
                    base_delay = queue_length / arrival_rate * 1000
                    
                    # 添加真实网络的随机波动（±20%）
                    noise_factor = np.random.uniform(0.8, 1.2)
                    
                    # 考虑网络拥塞的影响（当利用率高时，延迟波动更大）
                    if utilization > 0.7:
                        congestion_factor = 1.0 + (utilization - 0.7) * 0.3 * np.random.uniform(0.5, 1.5)
                    else:
                        congestion_factor = 1.0
                    
                    # 考虑丢包对延迟的影响（丢包会导致重传，增加延迟）
                    if packet_loss > 0:
                        retransmission_delay = packet_loss * 50.0 * np.random.uniform(0.5, 1.5)  # 重传延迟
                    else:
                        retransmission_delay = 0.0
                    
                    # 计算实际延迟（理论值 + 噪声 + 拥塞 + 重传）
                    delay = base_delay * noise_factor * congestion_factor + retransmission_delay
                    delay = max(0.0, delay)  # 确保非负
                else:
                    delay = 0.0
                
                # 计算吞吐量
                throughput = arrival_rate * (1 - packet_loss) * 8 / 1e6  # 转换为Mbps
                
                state = NetworkState(
                    timestamp=timestamp,
                    node_id=node_id,
                    queue_length=queue_length,
                    arrival_rate=arrival_rate,
                    service_rate=service_rate,
                    utilization=utilization,
                    packet_loss=packet_loss,
                    delay=delay,
                    throughput=throughput
                )
                states.append(state)
        
        return states
    
    def generate_training_data(
        self,
        num_nodes: int = 20,
        num_samples: int = 1000,
        feature_window: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成训练数据（用于MLP）
        
        Args:
            num_nodes: 节点数量
            num_samples: 样本数量
            feature_window: 特征窗口大小
            
        Returns:
            (特征矩阵, 标签向量)
        """
        states = self.generate_network_states(
            num_nodes=num_nodes,
            time_steps=num_samples + feature_window
        )
        
        X = []
        y = []
        
        for i in range(feature_window, len(states)):
            # 特征：过去feature_window个时间步的状态
            features = []
            for j in range(i - feature_window, i):
                state = states[j]
                features.extend([
                    state.arrival_rate,
                    state.service_rate,
                    state.utilization,
                    state.queue_length
                ])
            
            # 标签：当前时间步的延迟
            label = states[i].delay
            
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)

