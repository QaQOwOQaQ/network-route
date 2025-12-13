"""
能力底图构建结果可视化模块
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
from .scenario_generator import CapabilityNode, SpatialCell
import matplotlib.font_manager as fm


class CapabilityMapVisualizer:
    """能力底图可视化器"""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        初始化可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        
        # 配置中文字体
        chinese_fonts = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP', 
                        'Noto Sans CJK KR', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
                        'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        font_to_use = None
        
        for font in chinese_fonts:
            if font in available_fonts:
                font_to_use = font
                break
        
        if font_to_use:
            plt.rcParams['font.sans-serif'] = [font_to_use] + [f for f in plt.rcParams['font.sans-serif'] if f != font_to_use]
        else:
            noto_fonts = [f for f in available_fonts if 'Noto' in f and 'CJK' in f]
            if noto_fonts:
                font_to_use = noto_fonts[0]
                plt.rcParams['font.sans-serif'] = [font_to_use] + plt.rcParams['font.sans-serif']
        
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_contact_graph(
        self,
        nodes: List[CapabilityNode],
        graph: nx.Graph,
        title: str = "接触图 (CGR)",
        save_path: Optional[str] = None
    ):
        """
        绘制接触图
        
        Args:
            nodes: 节点列表
            graph: 接触图
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        pos = {node.node_id: node.position for node in nodes}
        
        # 绘制边
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.4,
            width=1.5,
            edge_color='gray',
            ax=ax
        )
        
        # 绘制节点（按能力着色）
        node_capabilities = [node.capability for node in nodes]
        max_cap = max(node_capabilities) if node_capabilities else 1.0
        node_colors = [cap / max_cap for cap in node_capabilities]
        
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=400,
            cmap=plt.cm.Reds,
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
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_spatial_grid(
        self,
        nodes: List[CapabilityNode],
        spatial_grid: List[SpatialCell],
        area_size: Tuple[float, float],
        title: str = "空间网格索引",
        save_path: Optional[str] = None
    ):
        """
        绘制空间网格索引
        
        Args:
            nodes: 节点列表
            spatial_grid: 空间网格单元列表
            area_size: 区域大小
            title: 标题
            save_path: 保存路径
        """
        from typing import Tuple
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制网格单元
        cell_capabilities = [cell.total_capability for cell in spatial_grid]
        max_cap = max(cell_capabilities) if cell_capabilities else 1.0
        
        for cell in spatial_grid:
            x_min, y_min, x_max, y_max = cell.bounds
            width = x_max - x_min
            height = y_max - y_min
            
            # 按能力着色
            color_intensity = cell.total_capability / max_cap if max_cap > 0 else 0
            rect = plt.Rectangle(
                (x_min, y_min), width, height,
                facecolor=plt.cm.Blues(color_intensity),
                edgecolor='black',
                linewidth=0.5,
                alpha=0.6
            )
            ax.add_patch(rect)
            
            # 添加能力值标签
            if cell.total_capability > 0:
                ax.text((x_min + x_max) / 2, (y_min + y_max) / 2,
                       f'{cell.total_capability:.0f}',
                       ha='center', va='center',
                       fontsize=7, fontweight='bold')
        
        # 绘制节点
        node_positions = {node.node_id: node.position for node in nodes}
        for node_id, (x, y) in node_positions.items():
            ax.plot(x, y, 'ro', markersize=8, alpha=0.8)
            ax.text(x, y + 20, str(node_id), ha='center', fontsize=7, fontweight='bold')
        
        ax.set_xlim(0, area_size[0])
        ax.set_ylim(0, area_size[1])
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_capability_comparison(
        self,
        comparison_results: Dict[str, Dict],
        title: str = "能力底图构建算法对比",
        save_path: Optional[str] = None
    ):
        """
        绘制能力底图构建算法对比
        
        Args:
            comparison_results: 对比结果
            title: 标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        algo_names = list(comparison_results.keys())
        
        # 左图：成功率和其他指标
        ax1 = axes[0]
        metrics = ['success_rate', 'avg_capability', 'avg_bandwidth']
        metric_names = ['成功率', '平均能力', '平均带宽']
        
        x = np.arange(len(algo_names))
        width = 0.25
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [comparison_results[algo].get(metric, 0) for algo in algo_names]
            ax1.bar(x + i * width, values, width, label=name, alpha=0.8)
        
        ax1.set_xlabel('算法', fontsize=12)
        ax1.set_ylabel('指标值', fontsize=12)
        ax1.set_title('性能指标对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(algo_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 右图：路径长度和接触数
        ax2 = axes[1]
        metrics2 = ['avg_path_length', 'avg_contact_count']
        metric_names2 = ['平均路径长度', '平均接触数']
        
        for i, (metric, name) in enumerate(zip(metrics2, metric_names2)):
            values = [comparison_results[algo].get(metric, 0) for algo in algo_names]
            ax2.bar(x + i * width, values, width, label=name, alpha=0.8)
        
        ax2.set_xlabel('算法', fontsize=12)
        ax2.set_ylabel('指标值', fontsize=12)
        ax2.set_title('路径特征对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(algo_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

