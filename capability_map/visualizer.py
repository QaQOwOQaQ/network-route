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
        
        # 左图：能力相关指标（分别显示每个算法的指标）
        ax1 = axes[0]
        
        # 收集所有算法的能力相关指标
        all_capability_data = []
        all_capability_labels = []
        
        for algo in algo_names:
            algo_metrics = comparison_results[algo]
            if 'TEG/CGR建模' in algo:
                # TEG/CGR 指标
                if 'avg_capability' in algo_metrics:
                    all_capability_data.append(algo_metrics['avg_capability'])
                    all_capability_labels.append(f'{algo}\n平均能力')
                if 'success_rate' in algo_metrics:
                    all_capability_data.append(algo_metrics['success_rate'])
                    all_capability_labels.append(f'{algo}\n成功率')
                if 'avg_bandwidth' in algo_metrics:
                    all_capability_data.append(algo_metrics['avg_bandwidth'])
                    all_capability_labels.append(f'{algo}\n平均带宽')
            elif '空间网格索引' in algo:
                # 空间网格索引指标
                if 'avg_query_capability' in algo_metrics:
                    all_capability_data.append(algo_metrics['avg_query_capability'])
                    all_capability_labels.append(f'{algo}\n平均查询能力')
                if 'avg_cell_capability' in algo_metrics:
                    all_capability_data.append(algo_metrics['avg_cell_capability'])
                    all_capability_labels.append(f'{algo}\n平均网格能力')
                if 'avg_query_node_count' in algo_metrics:
                    all_capability_data.append(algo_metrics['avg_query_node_count'])
                    all_capability_labels.append(f'{algo}\n平均查询节点数')
        
        if all_capability_data:
            x1 = np.arange(len(all_capability_data))
            bars1 = ax1.bar(x1, all_capability_data, alpha=0.8, 
                           color=['#1f77b4' if 'TEG/CGR' in label else '#ff7f0e' for label in all_capability_labels])
            ax1.set_xticks(x1)
            ax1.set_xticklabels(all_capability_labels, rotation=45, ha='right', fontsize=9)
            # 添加数值标签
            for bar, val in zip(bars1, all_capability_data):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('算法指标', fontsize=12)
        ax1.set_ylabel('指标值', fontsize=12)
        ax1.set_title('性能指标对比', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 右图：结构特征对比（分别显示每个算法的结构指标）
        ax2 = axes[1]
        
        all_structure_data = []
        all_structure_labels = []
        
        for algo in algo_names:
            algo_metrics = comparison_results[algo]
            if 'TEG/CGR建模' in algo:
                # TEG/CGR 结构指标
                if 'avg_path_length' in algo_metrics:
                    all_structure_data.append(algo_metrics['avg_path_length'])
                    all_structure_labels.append(f'{algo}\n平均路径长度')
                if 'avg_contact_count' in algo_metrics:
                    all_structure_data.append(algo_metrics['avg_contact_count'])
                    all_structure_labels.append(f'{algo}\n平均接触数')
            elif '空间网格索引' in algo:
                # 空间网格索引结构指标
                if 'total_cells' in algo_metrics:
                    all_structure_data.append(algo_metrics['total_cells'])
                    all_structure_labels.append(f'{algo}\n总网格数')
                if 'max_cell_capability' in algo_metrics:
                    all_structure_data.append(algo_metrics['max_cell_capability'])
                    all_structure_labels.append(f'{algo}\n最大网格能力')
                if 'min_cell_capability' in algo_metrics:
                    all_structure_data.append(algo_metrics['min_cell_capability'])
                    all_structure_labels.append(f'{algo}\n最小网格能力')
        
        if all_structure_data:
            x2 = np.arange(len(all_structure_data))
            bars2 = ax2.bar(x2, all_structure_data, alpha=0.8,
                           color=['#1f77b4' if 'TEG/CGR' in label else '#ff7f0e' for label in all_structure_labels])
            ax2.set_xticks(x2)
            ax2.set_xticklabels(all_structure_labels, rotation=45, ha='right', fontsize=9)
            # 添加数值标签
            for bar, val in zip(bars2, all_structure_data):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('算法指标', fontsize=12)
        ax2.set_ylabel('指标值', fontsize=12)
        ax2.set_title('结构特征对比', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

