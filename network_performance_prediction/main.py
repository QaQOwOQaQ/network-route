"""
网络性能预测算法主程序
整合场景生成、算法实现、性能评估和可视化
"""

import os
import warnings
import numpy as np
from typing import Dict

# 全局抑制 matplotlib 字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*CJK.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
from .scenario_generator import PerformanceScenarioGenerator
from .queueing_theory import QueueingTheoryPredictor
from .mlp_predictor import MLPPredictor
from .performance_evaluator import PerformancePredictionEvaluator
from .visualizer import PerformancePredictionVisualizer


class NetworkPerformancePredictionPlatform:
    """网络性能预测算法对比验证平台"""
    
    def __init__(self, output_dir: str = "output/network_performance"):
        """
        初始化平台
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.scenario_generator = PerformanceScenarioGenerator(seed=42)
        self.evaluator = PerformancePredictionEvaluator()
        self.visualizer = PerformancePredictionVisualizer()
        
        self.states = None
        self.algorithms = {}
        self.results = {}
    
    def generate_scenario(
        self,
        num_nodes: int = 20,
        time_steps: int = 200,
        time_interval: float = 1.0
    ):
        """
        生成场景
        
        Args:
            num_nodes: 节点数量
            time_steps: 时间步数
            time_interval: 时间间隔
        """
        print("正在生成网络性能预测场景...")
        self.states = self.scenario_generator.generate_network_states(
            num_nodes=num_nodes,
            time_steps=time_steps,
            time_interval=time_interval
        )
        print(f"场景生成完成: {num_nodes}个节点, {time_steps}个时间步")
    
    def initialize_algorithms(self):
        """初始化所有算法"""
        print("正在初始化算法...")
        
        # 初始化2种性能预测算法
        self.algorithms['排队论'] = QueueingTheoryPredictor()
        
        # MLP需要训练数据
        print("  训练MLP模型...")
        X, y = self.scenario_generator.generate_training_data(
            num_nodes=20,
            num_samples=500,
            feature_window=10
        )
        mlp = MLPPredictor(hidden_layers=(100, 50), max_iter=500)
        train_results = mlp.train(X, y)
        print(f"  MLP训练完成: 测试集R²={train_results['test_score']:.4f}")
        self.algorithms['MLP'] = mlp
        
        print(f"算法初始化完成: {list(self.algorithms.keys())}")
    
    def run_algorithms(self):
        """运行所有算法"""
        if self.states is None:
            raise ValueError("请先生成场景")
        
        print("正在运行算法...")
        
        self.results = {}
        for algo_name, algorithm in self.algorithms.items():
            print(f"  运行 {algo_name} 算法...")
            
            if algo_name == '排队论':
                predictions = algorithm.predict_batch(self.states, model_type='mm1')
            elif algo_name == 'MLP':
                predictions = algorithm.predict_batch(self.states, feature_window=10)
            else:
                predictions = []
            
            self.results[algo_name] = predictions
        
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
        
        comparison = self.evaluator.compare_algorithms(self.results, self.states)
        
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
        if self.states is None:
            raise ValueError("请先生成场景")
        
        print("正在生成可视化结果...")
        
        # 1. 绘制预测对比图
        pred_path = os.path.join(self.output_dir, "性能预测对比.png")
        self.visualizer.plot_prediction_comparison(
            self.results,
            self.states,
            title="网络性能预测对比",
            save_path=pred_path
        )
        print(f"  性能预测对比图已保存: {pred_path}")
        
        # 2. 绘制性能指标对比
        if comparison_results is None:
            comparison_results = self.evaluate_performance()
        
        metrics_path = os.path.join(self.output_dir, "性能指标对比.png")
        self.visualizer.plot_performance_metrics(
            comparison_results,
            title="算法性能指标对比",
            save_path=metrics_path
        )
        print(f"  性能指标对比图已保存: {metrics_path}")
        
        print("可视化完成")


def main():
    """主函数"""
    import numpy as np
    
    print("=" * 60)
    print("网络性能预测算法对比验证仿真平台")
    print("=" * 60)
    
    # 创建平台
    platform = NetworkPerformancePredictionPlatform(output_dir="output/network_performance")
    
    # 1. 生成场景
    platform.generate_scenario(
        num_nodes=20,
        time_steps=200,
        time_interval=1.0
    )
    
    # 2. 初始化算法
    platform.initialize_algorithms()
    
    # 3. 运行算法
    platform.run_algorithms()
    
    # 4. 评估性能
    comparison_results = platform.evaluate_performance()
    
    # 5. 可视化结果
    platform.visualize_results(comparison_results)
    
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

