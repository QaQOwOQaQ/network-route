# 路径规划算法模块

本模块实现了路径规划算法的完整功能，包括场景生成、算法实现、性能评估和可视化。

## 功能模块

### 1. 场景想定和数据生成 (`scenario_generator.py`)

- **Node**: 网络节点，包含位置、容量、可用时间窗口
- **Link**: 网络链路，包含带宽、延迟、时间窗口、可靠性
- **Task**: 任务，包含源节点、目标节点、数据大小、截止时间、优先级
- **ScenarioGenerator**: 场景生成器，生成动态网络拓扑和任务

### 2. 算法实现

#### 2.1 CGR算法 (`cgr_algorithm.py`)
- **CGRAlgorithm**: 接触图路由算法
- 适用于延迟容忍网络(DTN)
- 考虑节点和链路的时间窗口约束

#### 2.2 Dijkstra和A*算法 (`dijkstra_astar_algorithm.py`)
- **DijkstraAlgorithm**: Dijkstra最短路径算法
- **AStarAlgorithm**: A*启发式搜索算法
- 支持不同的权重类型（延迟、代价）

#### 2.3 分布式路由算法 (`distributed_routing.py`)
- **DistributedRoutingAlgorithm**: 分布式路由算法
- 基于距离向量路由协议思想
- 使用Bellman-Ford算法构建路由表

### 3. 性能指标评估 (`performance_evaluator.py`)

- **PerformanceEvaluator**: 性能评估器
- 评估指标：
  - 成功率 (success_rate)
  - 平均延迟 (avg_delay)
  - 平均路径长度 (avg_path_length)
  - 截止时间满足率 (deadline_satisfaction_rate)
  - 时间余量 (time_margin)
  - 平均每跳延迟 (avg_delay_per_hop)

### 4. 结果可视化 (`visualizer.py`)

- **Visualizer**: 可视化器
- 功能：
  - 网络拓扑图
  - 路径对比图
  - 性能对比柱状图
  - 延迟分布图

## 使用方法

### 快速开始

```python
from path_planning.main import PathPlanningPlatform

# 创建平台
platform = PathPlanningPlatform(output_dir="output/path_planning")

# 1. 生成场景
platform.generate_scenario(
    num_nodes=20,
    num_tasks=50,
    area_size=(1000, 1000)
)

# 2. 初始化算法
platform.initialize_algorithms()

# 3. 运行算法
platform.run_algorithms(sample_tasks=50)

# 4. 评估性能
comparison_results = platform.evaluate_performance()

# 5. 可视化结果
platform.visualize_results(comparison_results)
```

### 命令行运行

```bash
python run_path_planning.py
```

## 输出文件

运行后会在 `output/` 目录下生成：

1. `network_topology.png` - 网络拓扑图
2. `path_comparison_task_*.png` - 各任务的路径对比图
3. `performance_comparison.png` - 算法性能对比图
4. `delay_distribution.png` - 延迟分布对比图

## 算法对比

平台实现了4种路径规划算法：

1. **CGR** - 接触图路由，适用于DTN网络
2. **Dijkstra** - 经典最短路径算法
3. **A*** - 启发式搜索算法
4. **Distributed** - 分布式路由算法

每种算法都会在相同的场景和任务下运行，并生成性能对比报告。

