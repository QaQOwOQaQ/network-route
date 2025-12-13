# 算法测试指南

本文档说明如何分别测试4类算法。

## 一、路径规划算法（3种）

### 测试方法1：使用运行脚本
```bash
python run_path_planning.py
```

### 测试方法2：使用模块方式
```bash
python -m path_planning.main
```

### 测试方法3：使用Python交互式
```python
from path_planning.main import PathPlanningPlatform

platform = PathPlanningPlatform(output_dir="output")
platform.generate_scenario(num_nodes=40, num_tasks=50, area_size=(2000, 2000))
platform.initialize_algorithms()
platform.run_algorithms(sample_tasks=50)
comparison_results = platform.evaluate_performance()
platform.visualize_results(comparison_results)
```

### 输出文件
- `output/网络拓扑图.png`
- `output/路径对比_任务*.png`
- `output/算法性能对比.png`
- `output/延迟分布对比.png`

---

## 二、网络性能预测算法（2种）

### 测试方法1：使用模块方式
```bash
python -m network_performance_prediction.main
```

### 测试方法2：使用Python交互式
```python
from network_performance_prediction.main import NetworkPerformancePredictionPlatform

platform = NetworkPerformancePredictionPlatform(output_dir="output/network_performance")
platform.generate_scenario(num_nodes=20, time_steps=200, time_interval=1.0)
platform.initialize_algorithms()
platform.run_algorithms()
comparison_results = platform.evaluate_performance()
platform.visualize_results(comparison_results)
```

### 输出文件
- `output/network_performance/性能预测对比.png`
- `output/network_performance/性能指标对比.png`

---

## 三、节点优化排序算法（2种）

### 测试方法1：使用模块方式
```bash
python -m node_optimization.main
```

### 测试方法2：使用Python交互式
```python
from node_optimization.main import NodeOptimizationPlatform

platform = NodeOptimizationPlatform(output_dir="output/node_optimization")
platform.generate_scenario(num_nodes=20, num_tasks=50, num_controllers=3, area_size=(1000, 1000))
platform.initialize_algorithms()
platform.run_algorithms(num_tasks=50)
comparison_results = platform.evaluate_performance()
platform.visualize_results(comparison_results)
```

### 输出文件
- `output/node_optimization/节点任务分布图.png`
- `output/node_optimization/节点优化算法对比.png`

---

## 四、能力底图构建算法（2种）

### 测试方法1：使用模块方式
```bash
python -m capability_map.main
```

### 测试方法2：使用Python交互式
```python
from capability_map.main import CapabilityMapPlatform

platform = CapabilityMapPlatform(output_dir="output/capability_map")
platform.generate_scenario(num_nodes=30, area_size=(1000, 1000), time_horizon=200.0, grid_size=(10, 10))
platform.initialize_algorithms()
platform.run_algorithms(num_queries=20)
comparison_results = platform.evaluate_performance()
platform.visualize_results(comparison_results)
```

### 输出文件
- `output/capability_map/接触图_CGR建模.png`
- `output/capability_map/空间网格索引.png`
- `output/capability_map/能力底图构建算法对比.png`

---

## 快速测试脚本

创建快速测试脚本，可以分别测试每个模块：

```bash
# 测试路径规划
python run_path_planning.py

# 测试网络性能预测
python -m network_performance_prediction.main

# 测试节点优化排序
python -m node_optimization.main

# 测试能力底图构建
python -m capability_map.main
```

---

## 运行所有算法

如果想一次性运行所有4类算法：

```bash
python run_all_algorithms.py
```

---

## 测试参数调整

每个模块都支持参数调整，可以在对应的 `main.py` 中修改：

1. **路径规划算法**: 节点数、任务数、区域大小
2. **网络性能预测算法**: 节点数、时间步数
3. **节点优化排序算法**: 节点数、任务数、SDN控制器数
4. **能力底图构建算法**: 节点数、区域大小、网格大小

