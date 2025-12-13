"""
强化学习节点优化排序算法
使用Q-learning进行节点选择优化
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .scenario_generator import Node, Task


class RLNodeOptimizer:
    """强化学习节点优化器"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, epsilon: float = 0.1):
        """
        初始化强化学习优化器
        
        Args:
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # Q表: {(state, action): Q_value}
        self.episode_count = 0
    
    def _get_state(self, node: Node, task: Task) -> Tuple:
        """
        获取状态表示
        
        Args:
            node: 节点
            task: 任务
            
        Returns:
            状态元组
        """
        # 状态：节点容量利用率、内存利用率、带宽利用率、能量水平、任务优先级
        capacity_util = node.current_load
        memory_util = 0.5  # 简化
        bandwidth_util = 0.5  # 简化
        energy_level = node.energy_level
        task_priority = task.priority / 5.0  # 归一化
        
        # 离散化状态
        capacity_bin = int(capacity_util * 5)  # 0-4
        energy_bin = int(energy_level * 5)  # 0-4
        priority_bin = task_priority - 1  # 0-4
        
        return (capacity_bin, energy_bin, priority_bin)
    
    def _get_reward(self, node: Node, task: Task) -> float:
        """
        计算奖励
        
        Args:
            node: 节点
            task: 任务
            
        Returns:
            奖励值
        """
        # 检查资源是否足够
        if (node.capacity * (1 - node.current_load) < task.required_capacity or
            node.memory < task.required_memory or
            node.bandwidth < task.required_bandwidth):
            return -100.0  # 资源不足，大惩罚
        
        # 奖励：基于资源利用率、能量水平、优先级
        capacity_available = node.capacity * (1 - node.current_load)
        utilization_score = capacity_available / task.required_capacity
        energy_score = node.energy_level
        priority_score = task.priority / 5.0
        
        reward = utilization_score * 10 + energy_score * 5 + priority_score * 3
        
        return reward
    
    def select_node(
        self,
        nodes: List[Node],
        task: Task,
        training: bool = False
    ) -> Optional[int]:
        """
        选择最优节点
        
        Args:
            nodes: 节点列表
            task: 任务
            training: 是否训练模式
            
        Returns:
            选中的节点ID，如果无可用节点则返回None
        """
        if len(nodes) == 0:
            return None
        
        # 过滤可用节点
        available_nodes = []
        for node in nodes:
            if (node.capacity * (1 - node.current_load) >= task.required_capacity and
                node.memory >= task.required_memory and
                node.bandwidth >= task.required_bandwidth):
                available_nodes.append(node)
        
        if len(available_nodes) == 0:
            return None
        
        # 使用epsilon-贪婪策略选择节点
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择
            selected_node = np.random.choice(available_nodes)
        else:
            # 利用：选择Q值最高的节点
            best_q = -float('inf')
            selected_node = available_nodes[0]
            
            for node in available_nodes:
                state = self._get_state(node, task)
                action = node.node_id
                
                q_value = self.q_table.get((state, action), 0.0)
                if q_value > best_q:
                    best_q = q_value
                    selected_node = node
        
        return selected_node.node_id
    
    def update_q_value(
        self,
        node: Node,
        task: Task,
        reward: float,
        next_state: Optional[Tuple] = None,
        next_action: Optional[int] = None
    ):
        """
        更新Q值
        
        Args:
            node: 节点
            task: 任务
            reward: 奖励
            next_state: 下一状态
            next_action: 下一动作
        """
        state = self._get_state(node, task)
        action = node.node_id
        
        # 获取当前Q值
        current_q = self.q_table.get((state, action), 0.0)
        
        # 计算下一状态的最大Q值
        if next_state is not None and next_action is not None:
            next_q = self.q_table.get((next_state, next_action), 0.0)
        else:
            next_q = 0.0
        
        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def optimize_ordering(
        self,
        nodes: List[Node],
        tasks: List[Task],
        num_episodes: int = 100
    ) -> List[Tuple[int, int]]:
        """
        优化节点-任务排序
        
        Args:
            nodes: 节点列表
            tasks: 任务列表
            num_episodes: 训练轮数
            
        Returns:
            [(任务ID, 节点ID)] 排序结果
        """
        print(f"  强化学习训练中... (共{num_episodes}轮)")
        
        # 训练
        for episode in range(num_episodes):
            # 随机打乱任务顺序
            shuffled_tasks = tasks.copy()
            np.random.shuffle(shuffled_tasks)
            
            for task in shuffled_tasks:
                selected_node_id = self.select_node(nodes, task, training=True)
                
                if selected_node_id is not None:
                    selected_node = next(n for n in nodes if n.node_id == selected_node_id)
                    reward = self._get_reward(selected_node, task)
                    
                    # 更新Q值
                    self.update_q_value(selected_node, task, reward)
            
            if (episode + 1) % 20 == 0:
                print(f"    训练进度: {episode + 1}/{num_episodes}")
        
        # 执行优化排序
        assignments = []
        for task in tasks:
            selected_node_id = self.select_node(nodes, task, training=False)
            if selected_node_id is not None:
                assignments.append((task.task_id, selected_node_id))
        
        return assignments

