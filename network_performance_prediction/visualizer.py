"""
网络性能预测结果可视化模块
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from .scenario_generator import NetworkState
import matplotlib.font_manager as fm


class PerformancePredictionVisualizer:
    """性能预测可视化器"""
    
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
    
    def plot_prediction_comparison(
        self,
        predictions: Dict[str, List[Dict[str, float]]],
        actual_states: List[NetworkState],
        title: str = "性能预测对比",
        save_path: Optional[str] = None
    ):
        """
        绘制预测结果对比
        
        Args:
            predictions: {算法名称: 预测结果列表}
            actual_states: 实际网络状态列表
            title: 标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        # 提取时间序列
        timestamps = [s.timestamp for s in actual_states]
        delays_actual = [s.delay for s in actual_states]
        
        # 绘制延迟预测对比
        ax1 = axes[0]
        ax1.plot(timestamps, delays_actual, 'k-', linewidth=2, label='实际延迟', alpha=0.7)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, (algo_name, pred_list) in enumerate(predictions.items()):
            delays_pred = [p.get('delay', 0) for p in pred_list]
            # 过滤无效值
            delays_pred = [d if d != float('inf') and not np.isnan(d) else 0 for d in delays_pred]
            ax1.plot(timestamps[:len(delays_pred)], delays_pred, 
                    color=colors[idx % len(colors)], linestyle='--', 
                    linewidth=1.5, label=f'{algo_name}预测', alpha=0.8)
        
        ax1.set_xlabel('时间 (s)', fontsize=12)
        ax1.set_ylabel('延迟 (ms)', fontsize=12)
        ax1.set_title('延迟预测对比', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 绘制误差对比
        ax2 = axes[1]
        algo_names = list(predictions.keys())
        errors = []
        
        for algo_name in algo_names:
            pred_list = predictions[algo_name]
            delays_pred = [p.get('delay', 0) for p in pred_list]
            delays_pred = [d if d != float('inf') and not np.isnan(d) else 0 for d in delays_pred]
            
            min_len = min(len(delays_pred), len(delays_actual))
            errors_algo = np.abs(np.array(delays_pred[:min_len]) - np.array(delays_actual[:min_len]))
            errors.append(np.mean(errors_algo))
        
        bars = ax2.bar(range(len(algo_names)), errors, alpha=0.7, 
                      color=colors[:len(algo_names)])
        
        ax2.set_xlabel('算法', fontsize=12)
        ax2.set_ylabel('平均绝对误差 (ms)', fontsize=12)
        ax2.set_title('预测误差对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(algo_names)))
        ax2.set_xticklabels(algo_names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_metrics(
        self,
        comparison_results: Dict[str, Dict],
        title: str = "算法性能指标对比",
        save_path: Optional[str] = None
    ):
        """
        绘制性能指标对比
        
        Args:
            comparison_results: {算法名称: 性能指标字典}
            title: 标题
            save_path: 保存路径
        """
        metrics = ['mae', 'rmse', 'mape', 'r2_score']
        metric_names = {
            'mae': '平均绝对误差 (ms)',
            'rmse': '均方根误差 (ms)',
            'mape': '平均绝对百分比误差 (%)',
            'r2_score': 'R² 分数'
        }
        
        algo_names = list(comparison_results.keys())
        num_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
        if num_metrics == 1:
            axes = [axes]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = []
            labels = []
            
            for algo_name in algo_names:
                value = comparison_results[algo_name].get(metric, 0)
                if metric == 'r2_score':
                    # R²分数越大越好，其他越小越好
                    values.append(value if value != -float('inf') else 0)
                else:
                    values.append(value if value != float('inf') else 0)
                labels.append(algo_name)
            
            bars = ax.bar(range(len(algo_names)), values, alpha=0.7,
                         color=colors[:len(algo_names)])
            
            ax.set_xlabel('算法', fontsize=12)
            ax.set_ylabel(metric_names[metric], fontsize=12)
            ax.set_title(metric_names[metric], fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(algo_names)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
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

