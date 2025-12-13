"""
能力底图构建算法主程序
整合场景生成、算法实现、性能评估和可视化
"""

import os
import numpy as np
from typing import Dict, List
from .scenario_generator import CapabilityMapScenarioGenerator
from .teg_cgr_modeling import TEGCGRModeling
from .spatial_hashing import SpatialHashingModeling
from .performance_evaluator import CapabilityMapEvaluator
from .visualizer import CapabilityMapVisualizer


class CapabilityMapPlatform:
    """能力底图构建算法对比验证平台"""
    
    def __init__(self, output_dir: str = "output/capability_map"):
        """
        初始化平台
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.scenario_generator = CapabilityMapScenarioGenerator(seed=42)
        self.evaluator = CapabilityMapEvaluator()
        self.visualizer = CapabilityMapVisualizer()
        
        self.scenario = None
        self.algorithms = {}
        self.results = {}
    
    def generate_scenario(
        self,
        num_nodes: int = 30,
        area_size: tuple = (1000, 1000),
        time_horizon: float = 200.0,
        grid_size: tuple = (10, 10)
    ):
        """
        生成场景
        
        Args:
            num_nodes: 节点数量
            area_size: 区域大小
            time_horizon: 时间范围
            grid_size: 网格大小
        """
        print("正在生成能力底图构建场景...")
        self.scenario = self.scenario_generator.generate_dynamic_scenario(
            num_nodes=num_nodes,
            area_size=area_size,
            time_horizon=time_horizon,
            grid_size=grid_size
        )
        print(f"场景生成完成: {num_nodes}个节点, 网格大小: {grid_size}")
    
    def initialize_algorithms(self):
        """初始化所有算法"""
        if self.scenario is None:
            raise ValueError("请先生成场景")
        
        print("正在初始化算法...")
        
        nodes = self.scenario['nodes']
        contacts = self.scenario['contacts']
        graph = self.scenario['graph']
        spatial_grid = self.scenario['spatial_grid']
        
        # 初始化2种能力底图构建算法
        self.algorithms['TEG/CGR建模'] = TEGCGRModeling(nodes, contacts, graph)
        self.algorithms['空间网格索引'] = SpatialHashingModeling(nodes, spatial_grid)
        
        print(f"算法初始化完成: {list(self.algorithms.keys())}")
    
    def run_algorithms(self, num_queries: int = 20):
        """
        运行所有算法
        
        Args:
            num_queries: 查询数量
        """
        if not self.algorithms:
            raise ValueError("请先初始化算法")
        
        print(f"正在运行算法，查询数量: {num_queries}...")
        
        nodes = self.scenario['nodes']
        area_size = self.scenario['area_size']
        grid_size = self.scenario['grid_size']
        time_horizon = self.scenario['time_horizon']
        
        # 生成查询
        queries = []
        for i in range(num_queries):
            source = np.random.choice([n.node_id for n in nodes])
            dest = np.random.choice([n.node_id for n in nodes if n.node_id != source])
            start_time = np.random.uniform(0, time_horizon * 0.5)
            end_time = start_time + np.random.uniform(20, time_horizon * 0.3)
            queries.append({
                'source': source,
                'destination': dest,
                'start_time': start_time,
                'end_time': end_time
            })
        
        self.results = {}
        for algo_name, algorithm in self.algorithms.items():
            print(f"  运行 {algo_name} 算法...")
            results = []
            
            if algo_name == 'TEG/CGR建模':
                for query in queries:
                    result = algorithm.query_capability(
                        query['source'],
                        query['destination'],
                        query['start_time'],
                        query['end_time']
                    )
                    results.append(result)
            elif algo_name == '空间网格索引':
                # 构建能力底图
                capability_map = algorithm.build_capability_map(
                    grid_size=grid_size,
                    area_size=area_size
                )
                
                # 执行查询
                for query in queries:
                    source_node = next(n for n in nodes if n.node_id == query['source'])
                    dest_node = next(n for n in nodes if n.node_id == query['destination'])
                    
                    # 查询区域能力
                    region_bounds = (
                        min(source_node.position[0], dest_node.position[0]) - 100,
                        min(source_node.position[1], dest_node.position[1]) - 100,
                        max(source_node.position[0], dest_node.position[0]) + 100,
                        max(source_node.position[1], dest_node.position[1]) + 100
                    )
                    
                    result = algorithm.query_capability_in_region(
                        region_bounds,
                        grid_size=grid_size,
                        area_size=area_size
                    )
                    results.append(result)
            
            self.results[algo_name] = results
        
        self.queries = queries
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
        
        comparison = {}
        
        # 评估TEG/CGR建模
        if 'TEG/CGR建模' in self.results:
            comparison['TEG/CGR建模'] = self.evaluator.evaluate_teg_cgr(
                self.results['TEG/CGR建模'],
                self.queries
            )
        
        # 评估空间网格索引
        if '空间网格索引' in self.results:
            spatial_algo = self.algorithms['空间网格索引']
            capability_map = spatial_algo.build_capability_map(
                grid_size=self.scenario['grid_size'],
                area_size=self.scenario['area_size']
            )
            comparison['空间网格索引'] = self.evaluator.evaluate_spatial_hashing(
                capability_map,
                self.results['空间网格索引']
            )
        
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
        contacts = self.scenario['contacts']
        graph = self.scenario['graph']
        spatial_grid = self.scenario['spatial_grid']
        area_size = self.scenario['area_size']
        
        # 1. 绘制接触图
        if 'TEG/CGR建模' in self.algorithms:
            cgr_path = os.path.join(self.output_dir, "接触图_CGR建模.png")
            self.visualizer.plot_contact_graph(
                nodes, graph,
                title="接触图 (CGR建模)",
                save_path=cgr_path
            )
            print(f"  接触图已保存: {cgr_path}")
        
        # 2. 绘制空间网格索引
        if '空间网格索引' in self.algorithms:
            grid_path = os.path.join(self.output_dir, "空间网格索引.png")
            self.visualizer.plot_spatial_grid(
                nodes, spatial_grid, area_size,
                title="空间网格索引/分区建模",
                save_path=grid_path
            )
            print(f"  空间网格索引图已保存: {grid_path}")
        
        # 3. 绘制性能对比
        if comparison_results is None:
            comparison_results = self.evaluate_performance()
        
        if comparison_results:
            comp_path = os.path.join(self.output_dir, "能力底图构建算法对比.png")
            self.visualizer.plot_capability_comparison(
                comparison_results,
                title="能力底图构建算法对比",
                save_path=comp_path
            )
            print(f"  算法对比图已保存: {comp_path}")
        
        print("可视化完成")


def main():
    """主函数"""
    print("=" * 60)
    print("能力底图构建算法对比验证仿真平台")
    print("=" * 60)
    
    # 创建平台
    platform = CapabilityMapPlatform(output_dir="output/capability_map")
    
    # 1. 生成场景
    platform.generate_scenario(
        num_nodes=30,
        area_size=(1000, 1000),
        time_horizon=200.0,
        grid_size=(10, 10)
    )
    
    # 2. 初始化算法
    platform.initialize_algorithms()
    
    # 3. 运行算法
    platform.run_algorithms(num_queries=20)
    
    # 4. 评估性能
    comparison_results = platform.evaluate_performance()
    
    # 5. 可视化结果
    platform.visualize_results(comparison_results)
    
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

