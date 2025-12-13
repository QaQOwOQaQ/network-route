#!/usr/bin/env python3
"""
运行所有算法模块的主程序
包括：路径规划、网络性能预测、节点优化排序、能力底图构建
"""

import sys
import os

def run_path_planning():
    """运行路径规划算法"""
    print("\n" + "=" * 70)
    print("1. 路径规划算法模块")
    print("=" * 70)
    from path_planning.main import main
    main()

def run_network_performance():
    """运行网络性能预测算法"""
    print("\n" + "=" * 70)
    print("2. 网络性能预测算法模块")
    print("=" * 70)
    from network_performance_prediction.main import main
    main()

def run_node_optimization():
    """运行节点优化排序算法"""
    print("\n" + "=" * 70)
    print("3. 节点优化排序算法模块")
    print("=" * 70)
    from node_optimization.main import main
    main()

def run_capability_map():
    """运行能力底图构建算法"""
    print("\n" + "=" * 70)
    print("4. 能力底图构建算法模块")
    print("=" * 70)
    from capability_map.main import main
    main()

def main():
    """主函数"""
    print("=" * 70)
    print("任务规划算法对比验证仿真平台 - 完整运行")
    print("=" * 70)
    print()
    print("本程序将依次运行以下4个算法模块：")
    print("  1. 路径规划算法（3种）")
    print("  2. 网络性能预测算法（2种）")
    print("  3. 节点优化排序算法（2种）")
    print("  4. 能力底图构建算法（2种）")
    print()
    
    try:
        # 1. 路径规划算法
        run_path_planning()
        
        # 2. 网络性能预测算法
        run_network_performance()
        
        # 3. 节点优化排序算法
        run_node_optimization()
        
        # 4. 能力底图构建算法
        run_capability_map()
        
        print("\n" + "=" * 70)
        print("所有算法模块运行完成！")
        print("=" * 70)
        print()
        print("输出文件位置：")
        print("  - 路径规划: output/")
        print("  - 网络性能预测: output/network_performance/")
        print("  - 节点优化排序: output/node_optimization/")
        print("  - 能力底图构建: output/capability_map/")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

