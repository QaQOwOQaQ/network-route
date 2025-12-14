"""
网络性能预测结果可视化模块
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings
from typing import List, Dict, Optional
from .scenario_generator import NetworkState
import matplotlib.font_manager as fm

# 抑制 matplotlib 字体警告 - 必须在导入 matplotlib 后立即设置
import os
# 设置环境变量，禁用 matplotlib 字体警告
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
# 禁用字体警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

# 抑制所有 matplotlib 相关的 UserWarning（特别是字体警告）
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', message='.*CJK.*')
warnings.filterwarnings('ignore', message='.*missing from font.*')
warnings.filterwarnings('ignore', message='.*font.*')

# 设置 matplotlib 日志级别，避免字体警告
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.text').setLevel(logging.ERROR)
logging.getLogger('matplotlib.backends').setLevel(logging.ERROR)


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
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 提取时间序列
        timestamps = np.array([s.timestamp for s in actual_states])
        delays_actual = np.array([s.delay for s in actual_states])
        
        # 绘制延迟预测对比
        ax1 = axes[0]
        
        # 收集所有有效数据点，用于确定y轴范围
        all_valid_delays = delays_actual[delays_actual >= 0].tolist()
        
        # 定义更好的颜色和线型（折线图样式）- 使用更鲜明的颜色
        algo_styles = {
            '排队论': {'color': '#d62728', 'linestyle': '-', 'linewidth': 2.5, 'marker': 'o', 'markersize': 4},
            'MLP': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2.5, 'marker': 's', 'markersize': 4},
        }
        
        colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
        
        # 先绘制实际延迟（作为背景，使用浅灰色虚线）
        ax1.plot(timestamps, delays_actual, color='#999999', linewidth=1.5, 
                linestyle='--', marker='',  # 不使用标记点，减少视觉干扰
                label='实际延迟', alpha=0.5, zorder=1)
        
        # 绘制每个算法的预测结果（更突出）
        for idx, (algo_name, pred_list) in enumerate(predictions.items()):
            # 提取预测延迟，过滤无效值
            delays_pred = []
            valid_timestamps = []
            valid_indices = []
            
            for i, pred in enumerate(pred_list):
                delay = pred.get('delay', None)
                if delay is not None and delay != float('inf') and not np.isnan(delay) and delay >= 0:
                    delays_pred.append(delay)
                    if i < len(timestamps):
                        valid_timestamps.append(timestamps[i])
                        valid_indices.append(i)
            
            if len(delays_pred) > 0:
                delays_pred = np.array(delays_pred)
                valid_timestamps = np.array(valid_timestamps)
                all_valid_delays.extend(delays_pred.tolist())
                
                # 使用算法特定的样式，如果没有则使用默认样式
                style = algo_styles.get(algo_name, {
                    'color': colors[idx % len(colors)],
                    'linestyle': '-',
                    'linewidth': 2.5,
                    'marker': 'o',
                    'markersize': 4
                })
                
                # 绘制折线图（带标记点，更突出）
                ax1.plot(valid_timestamps, delays_pred,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        linewidth=style['linewidth'],
                        marker=style['marker'],
                        markersize=style['markersize'],
                        markevery=max(1, len(valid_timestamps) // 50),  # 每隔一定点数显示一个标记，避免太密集
                        label=f'{algo_name}预测',
                        alpha=0.9,
                        zorder=3)  # 更高的zorder，确保预测线在最上层
        
        # 自动调整y轴范围：使用分位数排除异常值，聚焦大部分数据
        if all_valid_delays:
            all_valid_delays_array = np.array(all_valid_delays)
            
            # 使用95%分位数来确定y轴上限，排除极端异常值
            y_min = np.min(all_valid_delays_array)
            y_max_95 = np.percentile(all_valid_delays_array, 95)  # 95%分位数
            y_max_actual = np.max(all_valid_delays_array)
            
            # 如果最大值远大于95%分位数（超过2倍），说明有异常值，使用95%分位数
            # 否则使用实际最大值
            if y_max_actual > y_max_95 * 2:
                y_max = y_max_95
                # 在图上标注有数据超出显示范围
                ax1.text(0.98, 0.98, f'注：部分数据超出显示范围\n(最大值: {y_max_actual:.0f} ms)',
                        transform=ax1.transAxes, fontsize=9, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                y_max = y_max_actual
            
            y_range = y_max - y_min
            if y_range > 0:
                # 添加10%的边距
                y_margin = y_range * 0.1
                ax1.set_ylim(max(0, y_min - y_margin), y_max + y_margin)
            else:
                # 如果所有值相同，设置一个小的范围
                ax1.set_ylim(max(0, y_min - 10), y_max + 10)
        
        ax1.set_xlabel('时间 (s)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('延迟 (ms)', fontsize=13, fontweight='bold')
        ax1.set_title('延迟预测对比', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.set_facecolor('#fafafa')
        
        # 绘制误差对比
        ax2 = axes[1]
        algo_names = list(predictions.keys())
        errors = []
        error_labels = []
        
        for algo_name in algo_names:
            pred_list = predictions[algo_name]
            delays_pred = []
            delays_actual_matched = []
            
            # 匹配预测值和实际值
            for i, pred in enumerate(pred_list):
                if i < len(delays_actual):
                    delay = pred.get('delay', None)
                    if delay is not None and delay != float('inf') and not np.isnan(delay) and delay >= 0:
                        delays_pred.append(delay)
                        delays_actual_matched.append(delays_actual[i])
            
            if len(delays_pred) > 0:
                errors_algo = np.abs(np.array(delays_pred) - np.array(delays_actual_matched))
                errors.append(np.mean(errors_algo))
                error_labels.append(algo_name)
        
        if errors:
            bars = ax2.bar(range(len(error_labels)), errors, alpha=0.8, 
                          color=[algo_styles.get(name, {}).get('color', colors[i % len(colors)]) 
                                 for i, name in enumerate(error_labels)])
            
            ax2.set_xlabel('算法', fontsize=13, fontweight='bold')
            ax2.set_ylabel('平均绝对误差 (ms)', fontsize=13, fontweight='bold')
            ax2.set_title('预测误差对比', fontsize=15, fontweight='bold', pad=15)
            ax2.set_xticks(range(len(error_labels)))
            ax2.set_xticklabels(error_labels, rotation=0, ha='center', fontsize=11)
            ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax2.set_facecolor('#fafafa')
            
            # 添加数值标签
            for bar, error_val in zip(bars, errors):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{error_val:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle(title, fontsize=17, fontweight='bold', y=0.995)
        
        # 抑制字体警告
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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
            valid_indices = []  # 记录有效数据的索引
            
            for i, algo_name in enumerate(algo_names):
                value = comparison_results[algo_name].get(metric, None)
                
                # 处理不同的指标
                if metric == 'r2_score':
                    # R²分数：可以是负数（表示模型很差），但不能是 -inf
                    if value is not None and value != -float('inf') and not np.isnan(value) and not np.isinf(value):
                        values.append(value)
                        labels.append(algo_name)
                        valid_indices.append(i)
                    else:
                        # 如果值无效，跳过这个算法（不显示）
                        continue
                elif metric == 'mape':
                    # MAPE：如果是 inf 或 nan，跳过（表示无法计算）
                    if value is not None and value != float('inf') and not np.isnan(value) and not np.isinf(value):
                        values.append(value)
                        labels.append(algo_name)
                        valid_indices.append(i)
                    else:
                        # 如果值无效，跳过这个算法（不显示）
                        continue
                else:
                    # 其他指标（mae, rmse）：如果是 inf，跳过；但0值是有效的
                    if value is not None and value != float('inf') and not np.isnan(value) and not np.isinf(value):
                        values.append(value)
                        labels.append(algo_name)
                        valid_indices.append(i)
                    else:
                        # 如果值无效，跳过这个算法（不显示）
                        continue
            
            # 只有当有有效数据时才绘制
            if values:
                # 对于 R² 分数，如果存在负值，需要调整 y 轴范围
                if metric == 'r2_score' and any(v < 0 for v in values):
                    # 设置 y 轴范围，确保负值可见
                    y_min = min(values) * 1.1 if min(values) < 0 else 0
                    y_max = max(values) * 1.1 if max(values) > 0 else 0
                    ax.set_ylim(y_min, y_max)
                    # 添加 y=0 参考线
                    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
                
                bars = ax.bar(range(len(values)), values, alpha=0.7,
                             color=[colors[i % len(colors)] for i in valid_indices])
                
                ax.set_xlabel('算法', fontsize=12)
                ax.set_ylabel(metric_names[metric], fontsize=12)
                ax.set_title(metric_names[metric], fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
                
                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    # 根据值的正负调整标签位置
                    if height >= 0:
                        # 如果高度为0或很小，将标签放在柱状图上方一点
                        if height < 0.01:
                            label_y = max(height, 0.01) if max(values) > 0 else 0.01
                            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                                   f'{val:.3f}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')
                        else:
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.3f}',
                                   ha='center', va='bottom', fontsize=9)
                    else:
                        # 负值标签放在柱状图下方
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.3f}',
                               ha='center', va='top', fontsize=9)
                
                # 如果所有值都是0或很小，设置一个最小y轴范围以便显示
                if all(v == 0 or abs(v) < 0.01 for v in values):
                    ax.set_ylim(-0.1, max(0.1, max(values) + 0.1))
            else:
                # 如果没有有效数据，显示提示信息
                ax.text(0.5, 0.5, '无有效数据', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='gray')
                ax.set_xlabel('算法', fontsize=12)
                ax.set_ylabel(metric_names[metric], fontsize=12)
                ax.set_title(metric_names[metric], fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        # 抑制字体警告
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

