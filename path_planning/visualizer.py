"""
结果可视化模块
可视化网络拓扑、路径、性能对比等
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import warnings
from typing import List, Dict, Optional
from .scenario_generator import Node, Link, Task, ScenarioGenerator
from .cgr_algorithm import PathResult

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


class Visualizer:
    """可视化器"""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        初始化可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        # 配置中文字体，优先使用Noto Sans CJK（支持中文、日文、韩文）
        import matplotlib.font_manager as fm
        import os
        
        # 查找可用的中文字体（Noto CJK系列支持中文）
        chinese_fonts = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP', 
                        'Noto Sans CJK KR', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
                        'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        
        # 获取所有可用字体名称和路径
        available_fonts = {f.name: f.fname for f in fm.fontManager.ttflist}
        font_to_use = None
        
        # 优先查找简体中文版本
        for font in chinese_fonts:
            if font in available_fonts:
                font_to_use = font
                break
        
        # 如果没找到，尝试查找系统字体目录中的中文字体
        if not font_to_use:
            # 常见的系统字体路径
            font_dirs = [
                '/usr/share/fonts',
                '/usr/local/share/fonts',
                os.path.expanduser('~/.fonts'),
                '/System/Library/Fonts',  # macOS
            ]
            
            # 查找包含 CJK 或中文相关的字体文件
            for font_dir in font_dirs:
                if os.path.exists(font_dir):
                    for root, dirs, files in os.walk(font_dir):
                        for file in files:
                            if file.endswith(('.ttf', '.otf')) and any(
                                keyword in file.lower() for keyword in ['cjk', 'noto', 'wenquanyi', 'simhei']
                            ):
                                try:
                                    # 尝试加载字体
                                    font_path = os.path.join(root, file)
                                    font_prop = fm.FontProperties(fname=font_path)
                                    font_to_use = font_prop.get_name()
                                    break
                                except:
                                    continue
                        if font_to_use:
                            break
                if font_to_use:
                    break
        
        # 如果找到了字体，设置它
        if font_to_use:
            # 设置字体，确保中文字符能正确显示
            plt.rcParams['font.sans-serif'] = [font_to_use] + [f for f in plt.rcParams['font.sans-serif'] if f != font_to_use]
        else:
            # 如果仍然没找到，尝试使用任何包含 Noto 的字体
            noto_fonts = [f for f in available_fonts.keys() if 'Noto' in f]
            if noto_fonts:
                font_to_use = noto_fonts[0]
                plt.rcParams['font.sans-serif'] = [font_to_use] + plt.rcParams['font.sans-serif']
        
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    def plot_network_topology(
        self,
        nodes: List[Node],
        links: List[Link],
        graph: nx.Graph,
        title: str = "网络拓扑图",
        save_path: Optional[str] = None
    ):
        """
        绘制网络拓扑图
        
        Args:
            nodes: 节点列表
            links: 链路列表
            graph: 网络图
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 获取节点位置
        pos = {node.node_id: node.position for node in nodes}
        
        # 绘制边
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.3,
            width=1.5,
            edge_color='gray',
            ax=ax
        )
        
        # 绘制节点
        node_colors = [node.capacity / max(n.capacity for n in nodes) for node in nodes]
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=300,
            cmap=plt.cm.Blues,
            alpha=0.8,
            ax=ax
        )
        
        # 绘制节点标签
        nx.draw_networkx_labels(
            graph, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        # 添加图例说明
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.8, label='网络节点'),
            Patch(facecolor='gray', alpha=0.3, label='网络链路')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 抑制字体警告
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_paths_comparison(
        self,
        nodes: List[Node],
        links: List[Link],
        graph: nx.Graph,
        task: Task,
        paths: Dict[str, PathResult],
        title: str = "路径对比",
        save_path: Optional[str] = None
    ):
        """
        绘制多个算法的路径对比
        
        Args:
            nodes: 节点列表
            links: 链路列表
            graph: 网络图
            task: 任务
            paths: {算法名称: 路径结果}
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        pos = {node.node_id: node.position for node in nodes}
        
        # 绘制所有边（浅色）
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.1,
            width=0.5,
            edge_color='gray',
            ax=ax
        )
        
        # 为每个算法固定配色，避免和目标节点(red)冲突
        algo_colors = {
            'CGR': 'purple',        # CGR路径颜色：紫色（避免与红色目标节点冲突）
            'Dijkstra': 'blue',      # 蓝色
            'A*': 'green',          # 绿色
            'Distributed': 'orange'  # 橙色
        }
        
        # 为每个算法明确指定线型：主要算法用实线，次要的用虚线/点线区分
        algo_line_styles = {
            'CGR': '-',           # 紫色实线
            'Dijkstra': '--',     # 蓝色虚线
            'A*': '-',            # 绿色实线（连续直线）
            'Distributed': ':'    # 橙色点线
        }
        
        # 不同线宽，确保可见性
        algo_line_widths = {
            'CGR': 3.5,
            'Dijkstra': 3.0,
            'A*': 3.0,
            'Distributed': 2.5
        }
        
        # 算法中文名称映射
        algo_name_map = {
            'CGR': 'CGR算法',
            'Dijkstra': 'Dijkstra算法',
            'A*': 'A*算法',
            'Distributed': '分布式路由算法'
        }
        
        # 绘制每个算法的路径，使用不同的颜色和线型组合
        for algo_name, path_result in paths.items():
            if not path_result.feasible or len(path_result.path) < 2:
                continue
            
            # 获取中文算法名称
            display_name = algo_name_map.get(algo_name, algo_name)
            
            # 绘制路径边
            path_edges = [(path_result.path[i], path_result.path[i+1]) 
                         for i in range(len(path_result.path) - 1)]
            
            # 为每个算法分配不同的颜色和线型
            color = algo_colors.get(algo_name, 'black')
            line_style = algo_line_styles.get(algo_name, '-')
            line_width = algo_line_widths.get(algo_name, 2.5)
            
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=path_edges,
                edge_color=color,
                style=line_style,
                width=line_width,
                alpha=0.85,
                label=f"{display_name} (延迟: {path_result.total_delay:.2f}ms)",
                ax=ax
            )
        
        # 绘制所有节点
        nx.draw_networkx_nodes(
            graph, pos,
            node_color='lightblue',
            node_size=200,
            alpha=0.8,
            ax=ax
        )
        
        # 高亮源节点和目标节点
        source_node = [task.source]
        dest_node = [task.destination]
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=source_node,
            node_color='green',
            node_size=500,
            alpha=0.9,
            ax=ax
        )
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=dest_node,
            node_color='red',   # 目标节点保持红色
            node_size=500,
            alpha=0.9,
            ax=ax
        )
        
        # 绘制节点标签
        nx.draw_networkx_labels(
            graph, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        # 手动添加节点图例
        from matplotlib.patches import Patch
        node_legend = [
            Patch(facecolor='green', alpha=0.9, label='源节点'),
            Patch(facecolor='red', alpha=0.9, label='目标节点'),
            Patch(facecolor='lightblue', alpha=0.8, label='中间节点')
        ]
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        # 合并路径图例和节点图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + node_legend, loc='upper right', fontsize=9)
        ax.axis('off')
        
        # 抑制字体警告
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(
        self,
        comparison_results: Dict[str, Dict],
        metrics: List[str] = None,
        title: str = "算法性能对比",
        save_path: Optional[str] = None
    ):
        """
        绘制性能对比柱状图
        
        Args:
            comparison_results: {算法名称: 性能指标字典}
            metrics: 要对比的指标列表
            title: 标题
            save_path: 保存路径
        """
        if metrics is None:
            metrics = ['success_rate', 'avg_delay', 'avg_path_length', 'deadline_satisfaction_rate']
        
        # 指标中文名称映射
        metric_name_map = {
            'success_rate': '成功率',
            'avg_delay': '平均延迟(ms)',
            'avg_path_length': '平均路径长度',
            'deadline_satisfaction_rate': '截止时间满足率',
            'avg_cost': '平均代价',
            'avg_time_margin': '平均时间余量',
            'avg_delay_per_hop': '平均每跳延迟(ms)'
        }
        
        # 算法中文名称映射
        algo_name_map = {
            'CGR': 'CGR算法',
            'Dijkstra': 'Dijkstra算法',
            'A*': 'A*算法',
            'Distributed': '分布式路由算法'
        }
        
        # 过滤掉inf值
        filtered_results = {}
        for algo_name, metrics_dict in comparison_results.items():
            filtered_results[algo_name] = {
                k: v if v != float('inf') and not np.isnan(v) else 0.0
                for k, v in metrics_dict.items()
            }
        
        num_metrics = len(metrics)
        num_algorithms = len(filtered_results)
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
        if num_metrics == 1:
            axes = [axes]
        
        algo_names = list(filtered_results.keys())
        algo_display_names = [algo_name_map.get(name, name) for name in algo_names]
        x = np.arange(num_algorithms)
        width = 0.6
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [filtered_results[algo][metric] for algo in algo_names]
            
            bars = ax.bar(x, values, width, alpha=0.8)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            # 使用中文标签
            metric_display = metric_name_map.get(metric, metric.replace('_', ' '))
            ax.set_xlabel('算法', fontsize=12)
            ax.set_ylabel(metric_display, fontsize=12)
            ax.set_title(metric_display, fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(algo_display_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        # 抑制字体警告
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_delay_distribution(
        self,
        comparison_results: Dict[str, Dict],
        title: str = "延迟分布对比",
        save_path: Optional[str] = None
    ):
        """
        绘制延迟分布对比（需要原始数据，这里简化处理）
        
        Args:
            comparison_results: 对比结果
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 算法中文名称映射
        algo_name_map = {
            'CGR': 'CGR算法',
            'Dijkstra': 'Dijkstra算法',
            'A*': 'A*算法',
            'Distributed': '分布式路由算法'
        }
        
        algo_names = list(comparison_results.keys())
        delays = []
        labels = []
        
        for algo_name in algo_names:
            metrics = comparison_results[algo_name]
            avg_delay = metrics.get('avg_delay', 0)
            std_delay = metrics.get('std_delay', 0)
            
            if avg_delay != float('inf') and not np.isnan(avg_delay):
                delays.append(avg_delay)
                display_name = algo_name_map.get(algo_name, algo_name)
                labels.append(f"{display_name}\n(均值={avg_delay:.2f}ms, 标准差={std_delay:.2f}ms)")
        
        if delays:
            bars = ax.bar(range(len(delays)), delays, alpha=0.7, 
                         yerr=[comparison_results[algo].get('std_delay', 0) 
                               for algo in algo_names if algo in comparison_results],
                         capsize=5)
            
            ax.set_xlabel('算法', fontsize=12)
            ax.set_ylabel('平均延迟 (ms)', fontsize=12)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        # 抑制字体警告
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
