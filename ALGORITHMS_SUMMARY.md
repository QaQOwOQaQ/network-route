# 任务规划算法对比验证仿真平台 - 算法实现总结

## 已完成算法模块

### 一、路径规划算法（3种）✅

**模块路径**: `path_planning/`

1. **CGR算法** (Contact Graph Routing)
   - 文件: `cgr_algorithm.py`
   - 特点: 适用于延迟容忍网络，考虑时间窗口约束

2. **Dijkstra/A*算法**
   - 文件: `dijkstra_astar_algorithm.py`
   - 特点: 
     - Dijkstra: 最短路径算法，优化延迟
     - A*: 启发式搜索，考虑可靠性因子

3. **分布式路由算法** (Distributed Routing)
   - 文件: `distributed_routing.py`
   - 特点: 基于距离向量路由协议，优化综合代价

**运行方式**:
```bash
python run_path_planning.py
```

**输出目录**: `output/`

---

### 二、网络性能预测算法（2种）✅

**模块路径**: `network_performance_prediction/`

1. **排队论方法** (Queueing Theory)
   - 文件: `queueing_theory.py`
   - 特点: 基于M/M/1和M/M/c排队模型预测网络性能

2. **MLP方法** (Multi-Layer Perceptron)
   - 文件: `mlp_predictor.py`
   - 特点: 使用多层感知器进行性能预测

**运行方式**:
```bash
python -m network_performance_prediction.main
```

**输出目录**: `output/network_performance/`

---

### 三、节点优化排序算法（2种）✅

**模块路径**: `node_optimization/`

1. **强化学习方法** (Reinforcement Learning)
   - 文件: `reinforcement_learning.py`
   - 特点: 使用Q-learning进行节点选择优化

2. **SDN架构下的Dijkstra算法**
   - 文件: `sdn_dijkstra.py`
   - 特点: 在SDN控制器管理下进行节点优化排序

**运行方式**:
```bash
python -m node_optimization.main
```

**输出目录**: `output/node_optimization/`

---

### 四、能力底图构建算法（2种）✅

**模块路径**: `capability_map/`

1. **时间演进图/接触图路由建模** (TEG/CGR Modeling)
   - 文件: `teg_cgr_modeling.py`
   - 特点: 
     - 构建时间演进图 (Time-Expanded Graph)
     - 构建接触图 (Contact Graph)
     - 支持时间窗口查询

2. **空间网格索引/分区建模** (Spatial Hashing/Partitioning)
   - 文件: `spatial_hashing.py`
   - 特点:
     - 空间网格索引
     - 区域能力查询
     - 网络分区

**运行方式**:
```bash
python -m capability_map.main
```

**输出目录**: `output/capability_map/`

---

## 统一运行

运行所有算法模块：

```bash
python run_all_algorithms.py
```

这将依次运行所有4个算法模块，生成完整的对比验证结果。

---

## 每个算法模块包含的4个要素

所有算法模块都包含以下4个要素：

1. **场景想定、数据生成**
   - 每个模块都有 `scenario_generator.py`
   - 生成符合算法特点的测试场景和数据

2. **算法实现**
   - 每种算法都有独立的实现文件
   - 算法逻辑完整，参数可配置

3. **性能指标评估**
   - 每个模块都有 `performance_evaluator.py`
   - 评估算法性能指标（成功率、延迟、代价等）

4. **结果可视化**
   - 每个模块都有 `visualizer.py`
   - 生成中文标签的对比图表
   - 图片文件名使用中文

---

## 项目结构

```
network_route/
├── path_planning/              # 路径规划算法（3种）
│   ├── scenario_generator.py
│   ├── cgr_algorithm.py
│   ├── dijkstra_astar_algorithm.py
│   ├── distributed_routing.py
│   ├── performance_evaluator.py
│   ├── visualizer.py
│   └── main.py
│
├── network_performance_prediction/  # 网络性能预测算法（2种）
│   ├── scenario_generator.py
│   ├── queueing_theory.py
│   ├── mlp_predictor.py
│   ├── performance_evaluator.py
│   ├── visualizer.py
│   └── main.py
│
├── node_optimization/          # 节点优化排序算法（2种）
│   ├── scenario_generator.py
│   ├── reinforcement_learning.py
│   ├── sdn_dijkstra.py
│   ├── performance_evaluator.py
│   ├── visualizer.py
│   └── main.py
│
├── capability_map/             # 能力底图构建算法（2种）
│   ├── scenario_generator.py
│   ├── teg_cgr_modeling.py
│   ├── spatial_hashing.py
│   ├── performance_evaluator.py
│   ├── visualizer.py
│   └── main.py
│
├── run_path_planning.py       # 路径规划运行脚本
├── run_all_algorithms.py      # 运行所有算法
└── requirements.txt           # 依赖包
```

---

## 总计

- **路径规划算法**: 3种 ✅
- **网络性能预测算法**: 2种 ✅
- **节点优化排序算法**: 2种 ✅
- **能力底图构建算法**: 2种 ✅

**总计**: 9种算法，4个类别，全部完成！

