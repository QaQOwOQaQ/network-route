#!/usr/bin/env python3
"""
路径规划算法快速测试脚本
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_planning.main import PathPlanningPlatform

def test_basic():
    """基本功能测试"""
    print("开始测试路径规划算法...")
    
    try:
        # 创建平台
        platform = PathPlanningPlatform(output_dir="test_output")
        
        # 生成小规模场景用于测试
        print("\n1. 生成测试场景...")
        platform.generate_scenario(
            num_nodes=10,
            num_tasks=10,
            area_size=(500, 500)
        )
        
        # 初始化算法
        print("\n2. 初始化算法...")
        platform.initialize_algorithms()
        print(f"   已初始化算法: {list(platform.algorithms.keys())}")
        
        # 运行算法
        print("\n3. 运行算法...")
        platform.run_algorithms(sample_tasks=10)
        
        # 评估性能
        print("\n4. 评估性能...")
        comparison_results = platform.evaluate_performance()
        
        # 可视化（可选，测试时可能不需要）
        print("\n5. 生成可视化结果...")
        platform.visualize_results(comparison_results)
        
        print("\n✓ 测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic()
    sys.exit(0 if success else 1)

