"""
路径规划算法主程序
整合场景生成、算法实现、性能评估和可视化
"""

import os
import sys
import warnings
from typing import Dict, List
import numpy as np

# 全局抑制 matplotlib 字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*CJK.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from .scenario_generator import ScenarioGenerator
from .cgr_algorithm import CGRAlgorithm
from .dijkstra_astar_algorithm import DijkstraAlgorithm, AStarAlgorithm
from .distributed_routing import DistributedRoutingAlgorithm
from .performance_evaluator import PerformanceEvaluator
from .visualizer import Visualizer


class PathPlanningPlatform:
    """路径规划算法对比验证平台"""
    
    def __init__(self, output_dir: str = "output/path_planning"):
        """
        初始化平台
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.scenario_generator = ScenarioGenerator(seed=42)
        self.evaluator = PerformanceEvaluator()
        self.visualizer = Visualizer()
        
        self.scenario = None
        self.algorithms = {}
        self.results = {}
    
    def generate_scenario(
        self,
        num_nodes: int = 20,
        num_tasks: int = 50,
        area_size: tuple = (1000, 1000)
    ):
        """
        生成场景
        
        Args:
            num_nodes: 节点数量
            num_tasks: 任务数量
            area_size: 区域大小
        """
        print("正在生成场景...")
        self.scenario = self.scenario_generator.generate_dynamic_scenario(
            num_nodes=num_nodes,
            num_tasks=num_tasks,
            area_size=area_size
        )
        print(f"场景生成完成: {num_nodes}个节点, {num_tasks}个任务")
    
    def initialize_algorithms(self):
        """初始化所有算法"""
        if self.scenario is None:
            raise ValueError("请先生成场景")
        
        print("正在初始化算法...")
        
        nodes = self.scenario['nodes']
        links = self.scenario['links']
        graph = self.scenario['graph']
        
        # 初始化4种路径规划算法，使用不同的优化目标
        self.algorithms['CGR'] = CGRAlgorithm(nodes, links, graph)
        # Dijkstra: 优化延迟（最短延迟路径）
        self.algorithms['Dijkstra'] = DijkstraAlgorithm(nodes, links, graph)
        # A*: 优化延迟（使用启发式）
        self.algorithms['A*'] = AStarAlgorithm(nodes, links, graph)
        # Distributed: 优化综合代价（延迟+可靠性）
        self.algorithms['Distributed'] = DistributedRoutingAlgorithm(nodes, links, graph)
        
        print(f"算法初始化完成: {list(self.algorithms.keys())}")
    
    def run_algorithms(self, sample_tasks: int = None):
        """
        运行所有算法
        
        Args:
            sample_tasks: 采样任务数量（None表示使用所有任务）
        """
        if not self.algorithms:
            raise ValueError("请先初始化算法")
        
        tasks = self.scenario['tasks']
        if sample_tasks is not None:
            tasks = tasks[:sample_tasks]
        
        print(f"正在运行算法，任务数量: {len(tasks)}...")
        
        self.results = {}
        for algo_name, algorithm in self.algorithms.items():
            print(f"  运行 {algo_name} 算法...")
            results = []
            for idx, task in enumerate(tasks):
                # 为不同算法使用不同的优化目标，增加路径差异
                if algo_name == 'Dijkstra':
                    # Dijkstra: 优化延迟（最短延迟路径）
                    result = algorithm.find_path(task, current_time=0.0, weight_type='delay')
                elif algo_name == 'Distributed':
                    # Distributed: 使用综合代价（延迟+可靠性），通过fallback实现
                    result = algorithm.find_path(task, current_time=0.0)
                else:
                    # CGR和A*: 使用默认参数
                    result = algorithm.find_path(task, current_time=0.0)
                results.append(result)
                # 每10个任务显示一次进度
                if (idx + 1) % 10 == 0 or (idx + 1) == len(tasks):
                    print(f"    进度: {idx + 1}/{len(tasks)}")
            self.results[algo_name] = results
            print(f"  {algo_name} 算法完成")
        
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
        
        tasks = self.scenario['tasks']
        if len(self.results[list(self.results.keys())[0]]) < len(tasks):
            tasks = tasks[:len(self.results[list(self.results.keys())[0]])]
        
        comparison = self.evaluator.compare_algorithms(self.results, tasks)
        
        # 打印结果
        print("\n=== 性能评估结果 ===")
        for algo_name, metrics in comparison.items():
            print(f"\n{algo_name}:")
            for metric_name, value in metrics.items():
                if value != float('inf') and not np.isnan(value):
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
        links = self.scenario['links']
        graph = self.scenario['graph']
        
        # 1. 绘制网络拓扑
        topo_path = os.path.join(self.output_dir, "网络拓扑图.png")
        self.visualizer.plot_network_topology(
            nodes, links, graph,
            title="网络拓扑图",
            save_path=topo_path
        )
        print(f"  网络拓扑图已保存: {topo_path}")
        
        # 2. 绘制路径对比（优先选择算法路径有差异的任务）
        tasks = self.scenario['tasks']
        
        # 找出算法路径有差异的任务，优先选择
        candidate_tasks = []
        for task_idx in range(len(tasks)):
            task = tasks[task_idx]
            paths = {}
            for algo_name, results in self.results.items():
                if task_idx < len(results):
                    result = results[task_idx]
                    if result.feasible and len(result.path) >= 2:
                        paths[algo_name] = result.path
            
            if len(paths) >= 2:
                # 检查路径是否不同
                path_strings = {algo: '->'.join(map(str, path)) for algo, path in paths.items()}
                unique_paths = len(set(path_strings.values()))
                
                # 计算包含的算法数量（优先选择包含CGR的任务）
                has_cgr = 'CGR' in paths
                num_algorithms = len(paths)
                
                candidate_tasks.append({
                    'task_idx': task_idx,
                    'task': task,
                    'unique_paths': unique_paths,
                    'has_cgr': has_cgr,
                    'num_algorithms': num_algorithms,
                    'paths': paths
                })
        
        # 排序：优先选择路径差异大、包含CGR、算法数量多的任务
        # 注意：reverse=True时，True排在False前面，所以用not has_cgr来让True在前
        candidate_tasks.sort(key=lambda x: (
            -x['unique_paths'],      # 路径差异数（降序，差异越大越好）
            not x['has_cgr'],        # 包含CGR（False在前，即True优先）
            -x['num_algorithms']     # 算法数量（降序，算法越多越好）
        ))
        
        # 选择前3个最好的任务
        selected_tasks = candidate_tasks[:3]
        
        if not selected_tasks:
            # 如果没有找到差异任务，使用默认选择
            print("  警告: 未找到算法路径有差异的任务，使用默认选择")
            sample_indices = [0, len(tasks)//4, len(tasks)//2]
            selected_tasks = [{'task_idx': idx, 'task': tasks[idx], 'paths': {}} 
                            for idx in sample_indices if idx < len(tasks)]
        
        for task_info in selected_tasks:
            task_idx = task_info['task_idx']
            task = task_info['task']
            paths = {}
            for algo_name, results in self.results.items():
                if task_idx < len(results):
                    paths[algo_name] = results[task_idx]
            
            # 只绘制有可行路径的任务
            feasible_paths = {k: v for k, v in paths.items() 
                            if v.feasible and len(v.path) >= 2}
            
            if len(feasible_paths) >= 2:  # 至少需要2个算法有可行路径
                path_path = os.path.join(self.output_dir, f"路径对比_任务{task.task_id}.png")
                self.visualizer.plot_paths_comparison(
                    nodes, links, graph, task, feasible_paths,
                    title=f"任务 {task.task_id} 路径对比 (源节点: {task.source} -> 目标节点: {task.destination})",
                    save_path=path_path
                )
                print(f"  路径对比图已保存: {path_path} (包含算法: {list(feasible_paths.keys())})")
        
        # 3. 绘制性能对比
        if comparison_results is None:
            comparison_results = self.evaluate_performance()
        
        perf_path = os.path.join(self.output_dir, "算法性能对比.png")
        self.visualizer.plot_performance_comparison(
            comparison_results,
            metrics=['success_rate', 'avg_delay', 'avg_path_length', 'deadline_satisfaction_rate'],
            title="算法性能对比",
            save_path=perf_path
        )
        print(f"  性能对比图已保存: {perf_path}")
        
        # 4. 绘制延迟分布
        delay_path = os.path.join(self.output_dir, "延迟分布对比.png")
        self.visualizer.plot_delay_distribution(
            comparison_results,
            title="延迟分布对比",
            save_path=delay_path
        )
        print(f"  延迟分布图已保存: {delay_path}")
        
        print("可视化完成")


def main():
    """主函数"""
    print("=" * 60)
    print("路径规划算法对比验证仿真平台")
    print("=" * 60)
    
    # 创建平台
    platform = PathPlanningPlatform(output_dir="output/path_planning")
    
    # 1. 生成场景（增加复杂度，使路径更长更复杂）
    platform.generate_scenario(
        num_nodes=40,  # 增加节点数：20 -> 40，增加网络规模
        num_tasks=50,
        area_size=(2000, 2000)  # 扩大区域：1000x1000 -> 2000x2000，使节点更分散，路径更长
    )
    
    # 2. 初始化算法
    platform.initialize_algorithms()
    
    # 3. 运行算法
    platform.run_algorithms(sample_tasks=50)  # 使用所有任务
    
    # 4. 评估性能
    comparison_results = platform.evaluate_performance()
    
    # 5. 可视化结果
    platform.visualize_results(comparison_results)
    
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

