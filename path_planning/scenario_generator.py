"""
场景想定和数据生成模块
生成动态路由网络的场景数据，包括节点、链路、任务等
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random


@dataclass
class Node:
    """网络节点"""
    node_id: int
    position: Tuple[float, float]  # (x, y) 坐标
    capacity: float  # 节点容量
    available_time: float  # 可用时间窗口开始
    unavailable_time: float  # 不可用时间窗口结束


@dataclass
class Link:
    """网络链路"""
    source: int
    target: int
    bandwidth: float  # 带宽 (Mbps)
    delay: float  # 延迟 (ms)
    start_time: float  # 链路可用开始时间
    end_time: float  # 链路可用结束时间
    reliability: float  # 可靠性 (0-1)


@dataclass
class Task:
    """任务"""
    task_id: int
    source: int  # 源节点
    destination: int  # 目标节点
    data_size: float  # 数据大小 (MB)
    deadline: float  # 截止时间
    priority: int  # 优先级 (1-5, 5最高)


class ScenarioGenerator:
    """场景生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化场景生成器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_network_topology(
        self,
        num_nodes: int = 20,
        area_size: Tuple[float, float] = (1000, 1000),
        connection_probability: float = 0.2,  # 降低连接概率：0.3 -> 0.2，使网络更稀疏，路径更长
        min_bandwidth: float = 10.0,
        max_bandwidth: float = 100.0,
        min_delay: float = 1.0,
        max_delay: float = 50.0
    ) -> Tuple[List[Node], List[Link], nx.Graph]:
        """
        生成网络拓扑
        
        Args:
            num_nodes: 节点数量
            area_size: 区域大小 (width, height)
            connection_probability: 连接概率
            min_bandwidth: 最小带宽
            max_bandwidth: 最大带宽
            min_delay: 最小延迟
            max_delay: 最大延迟
            
        Returns:
            (节点列表, 链路列表, NetworkX图)
        """
        nodes = []
        links = []
        G = nx.Graph()
        
        # 生成节点
        for i in range(num_nodes):
            x = np.random.uniform(0, area_size[0])
            y = np.random.uniform(0, area_size[1])
            capacity = np.random.uniform(50, 200)
            # 节点可用时间窗口
            available_time = np.random.uniform(0, 100)
            unavailable_time = available_time + np.random.uniform(10, 50)
            
            node = Node(
                node_id=i,
                position=(x, y),
                capacity=capacity,
                available_time=available_time,
                unavailable_time=unavailable_time
            )
            nodes.append(node)
            G.add_node(i, pos=(x, y), capacity=capacity)
        
        # 生成链路（基于距离和概率）
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                pos_i = nodes[i].position
                pos_j = nodes[j].position
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                
                # 距离越近，连接概率越高（但整体降低，使网络更稀疏）
                max_distance = np.sqrt(area_size[0]**2 + area_size[1]**2)
                distance_factor = 1 - (distance / max_distance)
                # 调整概率计算，使网络更稀疏，路径更长
                prob = connection_probability * (0.2 + 0.5 * distance_factor)  # 降低基础概率
                
                if np.random.random() < prob:
                    bandwidth = np.random.uniform(min_bandwidth, max_bandwidth)
                    delay = np.random.uniform(min_delay, max_delay) + distance * 0.01
                    start_time = np.random.uniform(0, 50)
                    end_time = start_time + np.random.uniform(20, 100)
                    reliability = np.random.uniform(0.7, 0.99)
                    
                    link = Link(
                        source=i,
                        target=j,
                        bandwidth=bandwidth,
                        delay=delay,
                        start_time=start_time,
                        end_time=end_time,
                        reliability=reliability
                    )
                    links.append(link)
                    G.add_edge(i, j, 
                             bandwidth=bandwidth,
                             delay=delay,
                             start_time=start_time,
                             end_time=end_time,
                             reliability=reliability)
        
        return nodes, links, G
    
    def generate_tasks(
        self,
        nodes: List[Node],
        num_tasks: int = 50,
        min_data_size: float = 1.0,
        max_data_size: float = 100.0,
        time_horizon: float = 200.0
    ) -> List[Task]:
        """
        生成任务列表
        
        Args:
            nodes: 节点列表
            num_tasks: 任务数量
            min_data_size: 最小数据大小 (MB)
            max_data_size: 最大数据大小 (MB)
            time_horizon: 时间范围
            
        Returns:
            任务列表
        """
        tasks = []
        node_ids = [node.node_id for node in nodes]
        
        for i in range(num_tasks):
            source = np.random.choice(node_ids)
            destination = np.random.choice([n for n in node_ids if n != source])
            data_size = np.random.uniform(min_data_size, max_data_size)
            deadline = np.random.uniform(50, time_horizon)
            priority = np.random.randint(1, 6)
            
            task = Task(
                task_id=i,
                source=source,
                destination=destination,
                data_size=data_size,
                deadline=deadline,
                priority=priority
            )
            tasks.append(task)
        
        return tasks
    
    def generate_dynamic_scenario(
        self,
        num_nodes: int = 20,
        num_tasks: int = 50,
        time_steps: int = 10,
        area_size: Tuple[float, float] = (1000, 1000)
    ) -> Dict:
        """
        生成完整的动态场景
        
        Args:
            num_nodes: 节点数量
            num_tasks: 任务数量
            time_steps: 时间步数
            area_size: 区域大小
            
        Returns:
            包含所有场景数据的字典
        """
        nodes, links, graph = self.generate_network_topology(
            num_nodes=num_nodes,
            area_size=area_size
        )
        tasks = self.generate_tasks(nodes, num_tasks=num_tasks)
        
        scenario = {
            'nodes': nodes,
            'links': links,
            'graph': graph,
            'tasks': tasks,
            'num_nodes': num_nodes,
            'num_tasks': num_tasks,
            'time_steps': time_steps
        }
        
        return scenario

