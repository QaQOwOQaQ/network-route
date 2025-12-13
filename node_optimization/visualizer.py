"""
节点优化排序结果可视化模块
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
from .scenario_generator import Node, Task, SDNController
import matplotlib.font_manager as fm


class NodeOptimizationVisualizer:
    """节点优化排序可视化器"""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        初始化可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        
        # 配置中文字体
        chinese_fonts = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP', 
                        'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                break
        
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_node_task_distribution(
        self,
        nodes: List[Node],
        tasks: List[Task],
        controllers: Optional[List[SDNController]] = None,
        title: str = "节点和任务分布",
        save_path: Optional[str] = None
    ):
        """
        绘制节点和任务分布图（初始场景）
        
        Args:
            nodes: 节点列表
            tasks: 任务列表
            controllers: SDN控制器列表
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制节点
        for node in nodes:
            x, y = node.position
            # 节点大小表示资源容量
            size = 200 + node.capacity * 20
            # 颜色表示当前负载
            color_intensity = min(node.current_load, 1.0)
            color = plt.cm.Blues(color_intensity)
            ax.scatter(x, y, s=size, color=color, 
                      alpha=0.7, edgecolors='black', linewidths=1, label='节点')
            ax.text(x, y, f'N{node.node_id}', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 绘制任务（如果任务没有位置，则根据节点位置范围随机生成）
        if nodes:
            # 计算场景范围
            x_coords = [n.position[0] for n in nodes]
            y_coords = [n.position[1] for n in nodes]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 为任务生成随机位置（用于可视化）
            np.random.seed(42)  # 固定种子以保证可重复性
            for task in tasks:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                # 任务大小表示资源需求
                size = 100 + task.required_capacity * 5
                ax.scatter(x, y, s=size, c='red', marker='*', 
                          alpha=0.6, edgecolors='red', linewidths=1, label='任务')
                ax.text(x, y, f'T{task.task_id}', 
                       ha='center', va='center', fontsize=7, fontweight='bold', color='white')
        
        # 绘制SDN控制器
        if controllers:
            for ctrl in controllers:
                x, y = ctrl.position
                ax.scatter(x, y, s=500, c='green', marker='s', 
                          alpha=0.5, edgecolors='green', linewidths=2, label='SDN控制器')
                ax.text(x, y, f'C{ctrl.controller_id}', 
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')
                
                # 绘制控制器覆盖范围
                for node_id in ctrl.coverage_nodes:
                    node = next((n for n in nodes if n.node_id == node_id), None)
                    if node:
                        nx, ny = node.position
                        ax.plot([x, nx], [y, ny], 'g--', alpha=0.2, linewidth=1)
        
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 去重图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_node_task_assignment(
        self,
        nodes: List[Node],
        tasks: List[Task],
        assignments: List[Tuple[int, int]],
        controllers: Optional[List[SDNController]] = None,
        title: str = "节点任务分配",
        save_path: Optional[str] = None
    ):
        """
        绘制节点任务分配图
        
        Args:
            nodes: 节点列表
            tasks: 任务列表
            assignments: [(任务ID, 节点ID)] 分配结果
            controllers: SDN控制器列表
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制节点
        node_positions = {node.node_id: node.position for node in nodes}
        task_dict = {task.task_id: task for task in tasks}
        
        # 统计每个节点的任务数
        node_task_counts = {node.node_id: 0 for node in nodes}
        for task_id, node_id in assignments:
            if node_id in node_task_counts:
                node_task_counts[node_id] += 1
        
        # 绘制节点（大小表示任务数量）
        for node in nodes:
            x, y = node.position
            task_count = node_task_counts[node.node_id]
            size = 200 + task_count * 50
            
            # 颜色表示负载
            color_intensity = node.current_load
            ax.scatter(x, y, s=size, c=plt.cm.Reds(color_intensity), 
                      alpha=0.7, edgecolors='black', linewidths=1)
            ax.text(x, y, f'N{node.node_id}\n({task_count})', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 绘制SDN控制器
        if controllers:
            for ctrl in controllers:
                x, y = ctrl.position
                ax.scatter(x, y, s=500, c='blue', marker='s', 
                          alpha=0.5, edgecolors='blue', linewidths=2, label='SDN控制器')
                ax.text(x, y, f'C{ctrl.controller_id}', 
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(
        self,
        comparison_results: Dict[str, Dict],
        title: str = "算法性能对比",
        save_path: Optional[str] = None
    ):
        """
        绘制性能对比图
        
        Args:
            comparison_results: {算法名称: 性能指标字典}
            title: 标题
            save_path: 保存路径
        """
        metrics = ['success_rate', 'load_balance', 'avg_utilization', 
                  'priority_satisfaction', 'energy_efficiency']
        metric_names = {
            'success_rate': '成功率',
            'load_balance': '负载均衡度',
            'avg_utilization': '平均资源利用率',
            'priority_satisfaction': '优先级满足率',
            'energy_efficiency': '能量效率'
        }
        
        algo_names = list(comparison_results.keys())
        num_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
        if num_metrics == 1:
            axes = [axes]
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [comparison_results[algo].get(metric, 0) for algo in algo_names]
            
            bars = ax.bar(range(len(algo_names)), values, alpha=0.7,
                         color=colors[:len(algo_names)])
            
            ax.set_xlabel('算法', fontsize=12)
            ax.set_ylabel(metric_names[metric], fontsize=12)
            ax.set_title(metric_names[metric], fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(algo_names)))
            ax.set_xticklabels(algo_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

