"""
节点优化排序算法主程序
整合场景生成、算法实现、性能评估和可视化
"""

import os
import numpy as np
from typing import Dict
from .scenario_generator import NodeOptimizationScenarioGenerator
from .reinforcement_learning import RLNodeOptimizer
from .sdn_dijkstra import SDNDijkstraOptimizer
from .performance_evaluator import NodeOptimizationEvaluator
from .visualizer import NodeOptimizationVisualizer


class NodeOptimizationPlatform:
    """节点优化排序算法对比验证平台"""
    
    def __init__(self, output_dir: str = "output/node_optimization"):
        """
        初始化平台
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.scenario_generator = NodeOptimizationScenarioGenerator(seed=42)
        self.evaluator = NodeOptimizationEvaluator()
        self.visualizer = NodeOptimizationVisualizer()
        
        self.scenario = None
        self.algorithms = {}
        self.results = {}
    
    def generate_scenario(
        self,
        num_nodes: int = 20,
        num_tasks: int = 50,
        num_controllers: int = 3,
        area_size: tuple = (1000, 1000)
    ):
        """
        生成场景
        
        Args:
            num_nodes: 节点数量
            num_tasks: 任务数量
            num_controllers: SDN控制器数量
            area_size: 区域大小
        """
        print("正在生成节点优化排序场景...")
        self.scenario = self.scenario_generator.generate_scenario(
            num_nodes=num_nodes,
            num_tasks=num_tasks,
            num_controllers=num_controllers,
            area_size=area_size
        )
        print(f"场景生成完成: {num_nodes}个节点, {num_tasks}个任务, {num_controllers}个SDN控制器")
    
    def initialize_algorithms(self):
        """初始化所有算法"""
        if self.scenario is None:
            raise ValueError("请先生成场景")
        
        print("正在初始化算法...")
        
        nodes = self.scenario['nodes']
        tasks = self.scenario['tasks']
        controllers = self.scenario['controllers']
        
        # 初始化2种节点优化排序算法
        self.algorithms['强化学习'] = RLNodeOptimizer(
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1
        )
        self.algorithms['SDN_Dijkstra'] = SDNDijkstraOptimizer()
        
        print(f"算法初始化完成: {list(self.algorithms.keys())}")
    
    def run_algorithms(self, num_tasks: int = None):
        """
        运行所有算法
        
        Args:
            num_tasks: 任务数量（None表示使用所有任务）
        """
        if not self.algorithms:
            raise ValueError("请先初始化算法")
        
        tasks = self.scenario['tasks']
        if num_tasks is not None:
            tasks = tasks[:num_tasks]
        
        print(f"正在运行算法，任务数量: {len(tasks)}...")
        
        nodes = self.scenario['nodes']
        controllers = self.scenario['controllers']
        
        self.results = {}
        for algo_name, algorithm in self.algorithms.items():
            print(f"  运行 {algo_name} 算法...")
            results = []
            
            if algo_name == '强化学习':
                # 强化学习需要训练
                print("    训练强化学习模型...")
                for episode in range(50):  # 训练50个回合
                    for task in tasks[:10]:  # 使用前10个任务训练
                        # 选择节点（训练模式）
                        selected_node_id = algorithm.select_node(nodes, task, training=True)
                        if selected_node_id is not None:
                            selected_node = next(n for n in nodes if n.node_id == selected_node_id)
                            # 计算奖励
                            reward = algorithm._get_reward(selected_node, task)
                            algorithm.update_q_value(selected_node, task, reward)
                
                # 执行任务分配
                for task in tasks:
                    selected_node_id = algorithm.select_node(nodes, task, training=False)
                    if selected_node_id is not None:
                        selected_node = next(n for n in nodes if n.node_id == selected_node_id)
                        # 计算代价（使用奖励的负值）
                        cost = -algorithm._get_reward(selected_node, task)
                        results.append({
                            'task_id': task.task_id,
                            'selected_node': selected_node_id,
                            'feasible': True,
                            'cost': cost
                        })
                    else:
                        results.append({
                            'task_id': task.task_id,
                            'selected_node': None,
                            'feasible': False,
                            'cost': float('inf')
                        })
            
            elif algo_name == 'SDN_Dijkstra':
                # SDN架构下的Dijkstra算法
                # 使用optimize_ordering方法进行批量优化
                assignments = algorithm.optimize_ordering(nodes, tasks, controllers)
                assignment_dict = {task_id: node_id for task_id, node_id in assignments}
                
                for task in tasks:
                    if task.task_id in assignment_dict:
                        selected_node_id = assignment_dict[task.task_id]
                        selected_node = next(n for n in nodes if n.node_id == selected_node_id)
                        
                        # 找到管理该节点的控制器
                        controller = None
                        for ctrl in controllers:
                            if selected_node_id in ctrl.coverage_nodes:
                                controller = ctrl
                                break
                        
                        cost = algorithm._calculate_node_cost(selected_node, task, controller)
                        results.append({
                            'task_id': task.task_id,
                            'selected_node': selected_node_id,
                            'feasible': True,
                            'cost': cost
                        })
                    else:
                        results.append({
                            'task_id': task.task_id,
                            'selected_node': None,
                            'feasible': False,
                            'cost': float('inf')
                        })
            
            self.results[algo_name] = results
        
        print("算法运行完成")
    
    def evaluate_performance(self) -> Dict:
        """
        评估性能
        
        Returns:
            性能对比结果
        """
        if not self.results:
            raise ValueError("请先运行算法")
        
        print("正在评估性能...")
        
        nodes = self.scenario['nodes']
        tasks = self.scenario['tasks']
        
        # 转换结果格式：从 Dict 格式转换为 (task_id, node_id) 元组列表
        algorithm_assignments = {}
        for algo_name, results in self.results.items():
            assignments = []
            for result in results:
                if result.get('feasible') and result.get('selected_node') is not None:
                    assignments.append((result['task_id'], result['selected_node']))
            algorithm_assignments[algo_name] = assignments
        
        comparison = self.evaluator.compare_algorithms(algorithm_assignments, nodes, tasks)
        
        # 打印结果
        print("\n=== 性能评估结果 ===")
        for algo_name, metrics in comparison.items():
            print(f"\n{algo_name}:")
            for metric_name, value in metrics.items():
                if value != float('inf') and value != -float('inf') and not np.isnan(value):
                    print(f"  {metric_name}: {value:.4f}")
        
        return comparison
    
    def visualize_results(self, comparison_results: Dict = None):
        """
        可视化结果
        
        Args:
            comparison_results: 性能对比结果
        """
        if self.scenario is None:
            raise ValueError("请先生成场景")
        
        print("正在生成可视化结果...")
        
        nodes = self.scenario['nodes']
        tasks = self.scenario['tasks']
        controllers = self.scenario['controllers']
        
        # 1. 绘制节点和任务分布
        dist_path = os.path.join(self.output_dir, "节点任务分布图.png")
        self.visualizer.plot_node_task_distribution(
            nodes, tasks, controllers,
            title="节点和任务分布",
            save_path=dist_path
        )
        print(f"  节点任务分布图已保存: {dist_path}")
        
        # 2. 绘制性能对比
        if comparison_results is None:
            comparison_results = self.evaluate_performance()
        
        if comparison_results:
            perf_path = os.path.join(self.output_dir, "节点优化算法对比.png")
            self.visualizer.plot_performance_comparison(
                comparison_results,
                title="节点优化排序算法对比",
                save_path=perf_path
            )
            print(f"  算法对比图已保存: {perf_path}")
        
        print("可视化完成")


def main():
    """主函数"""
    print("=" * 60)
    print("节点优化排序算法对比验证仿真平台")
    print("=" * 60)
    
    # 创建平台
    platform = NodeOptimizationPlatform(output_dir="output/node_optimization")
    
    # 1. 生成场景
    platform.generate_scenario(
        num_nodes=20,
        num_tasks=50,
        num_controllers=3,
        area_size=(1000, 1000)
    )
    
    # 2. 初始化算法
    platform.initialize_algorithms()
    
    # 3. 运行算法
    platform.run_algorithms(num_tasks=50)
    
    # 4. 评估性能
    comparison_results = platform.evaluate_performance()
    
    # 5. 可视化结果
    platform.visualize_results(comparison_results)
    
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

