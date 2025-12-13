"""
节点优化排序场景生成模块
生成节点任务、资源、约束等数据
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random


@dataclass
class Node:
    """网络节点"""
    node_id: int
    position: Tuple[float, float]
    capacity: float  # 计算容量
    memory: float  # 内存容量
    bandwidth: float  # 带宽容量
    current_load: float  # 当前负载
    energy_level: float  # 能量水平


@dataclass
class Task:
    """任务"""
    task_id: int
    required_capacity: float  # 所需计算容量
    required_memory: float  # 所需内存
    required_bandwidth: float  # 所需带宽
    priority: int  # 优先级 (1-5)
    deadline: float  # 截止时间
    data_size: float  # 数据大小 (MB)


@dataclass
class SDNController:
    """SDN控制器"""
    controller_id: int
    position: Tuple[float, float]
    coverage_nodes: List[int]  # 覆盖的节点列表
    processing_capacity: float  # 处理能力


class NodeOptimizationScenarioGenerator:
    """节点优化排序场景生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化场景生成器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_nodes(
        self,
        num_nodes: int = 20,
        area_size: Tuple[float, float] = (1000, 1000)
    ) -> List[Node]:
        """
        生成节点
        
        Args:
            num_nodes: 节点数量
            area_size: 区域大小
            
        Returns:
            节点列表
        """
        nodes = []
        
        for i in range(num_nodes):
            x = np.random.uniform(0, area_size[0])
            y = np.random.uniform(0, area_size[1])
            capacity = np.random.uniform(50, 200)
            memory = np.random.uniform(100, 1000)  # MB
            bandwidth = np.random.uniform(10, 100)  # Mbps
            current_load = np.random.uniform(0, 0.7)
            energy_level = np.random.uniform(0.3, 1.0)
            
            node = Node(
                node_id=i,
                position=(x, y),
                capacity=capacity,
                memory=memory,
                bandwidth=bandwidth,
                current_load=current_load,
                energy_level=energy_level
            )
            nodes.append(node)
        
        return nodes
    
    def generate_tasks(
        self,
        num_tasks: int = 50,
        min_capacity: float = 10.0,
        max_capacity: float = 50.0
    ) -> List[Task]:
        """
        生成任务列表
        
        Args:
            num_tasks: 任务数量
            min_capacity: 最小计算需求
            max_capacity: 最大计算需求
            
        Returns:
            任务列表
        """
        tasks = []
        
        for i in range(num_tasks):
            required_capacity = np.random.uniform(min_capacity, max_capacity)
            required_memory = np.random.uniform(10, 100)  # MB
            required_bandwidth = np.random.uniform(1, 20)  # Mbps
            priority = np.random.randint(1, 6)
            deadline = np.random.uniform(50, 500)
            data_size = np.random.uniform(1, 100)  # MB
            
            task = Task(
                task_id=i,
                required_capacity=required_capacity,
                required_memory=required_memory,
                required_bandwidth=required_bandwidth,
                priority=priority,
                deadline=deadline,
                data_size=data_size
            )
            tasks.append(task)
        
        return tasks
    
    def generate_sdn_topology(
        self,
        nodes: List[Node],
        num_controllers: int = 3,
        coverage_radius: float = 400.0
    ) -> List[SDNController]:
        """
        生成SDN控制器拓扑
        
        Args:
            nodes: 节点列表
            num_controllers: 控制器数量
            coverage_radius: 覆盖半径
            
        Returns:
            SDN控制器列表
        """
        controllers = []
        area_size = (1000, 1000)  # 假设区域大小
        
        for i in range(num_controllers):
            x = np.random.uniform(0, area_size[0])
            y = np.random.uniform(0, area_size[1])
            
            # 计算覆盖的节点
            coverage_nodes = []
            for node in nodes:
                distance = np.sqrt((x - node.position[0])**2 + (y - node.position[1])**2)
                if distance <= coverage_radius:
                    coverage_nodes.append(node.node_id)
            
            processing_capacity = np.random.uniform(100, 500)
            
            controller = SDNController(
                controller_id=i,
                position=(x, y),
                coverage_nodes=coverage_nodes,
                processing_capacity=processing_capacity
            )
            controllers.append(controller)
        
        return controllers
    
    def generate_scenario(
        self,
        num_nodes: int = 20,
        num_tasks: int = 50,
        num_controllers: int = 3,
        area_size: Tuple[float, float] = (1000, 1000)
    ) -> Dict:
        """
        生成完整的节点优化排序场景
        
        Args:
            num_nodes: 节点数量
            num_tasks: 任务数量
            num_controllers: SDN控制器数量
            area_size: 区域大小
            
        Returns:
            包含所有场景数据的字典
        """
        nodes = self.generate_nodes(num_nodes, area_size)
        tasks = self.generate_tasks(num_tasks)
        controllers = self.generate_sdn_topology(nodes, num_controllers)
        
        scenario = {
            'nodes': nodes,
            'tasks': tasks,
            'controllers': controllers,
            'num_nodes': num_nodes,
            'num_tasks': num_tasks,
            'num_controllers': num_controllers,
            'area_size': area_size
        }
        
        return scenario

