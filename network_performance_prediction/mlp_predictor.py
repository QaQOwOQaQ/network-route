"""
MLP网络性能预测算法
使用多层感知器进行网络性能预测
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .scenario_generator import NetworkState


class MLPPredictor:
    """MLP性能预测器"""
    
    def __init__(self, hidden_layers: Tuple = (100, 50), max_iter: int = 500):
        """
        初始化MLP预测器
        
        Args:
            hidden_layers: 隐藏层结构
            max_iter: 最大迭代次数
        """
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        训练MLP模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练评估指标
        """
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # 创建并训练模型
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # 评估训练效果
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # 预测并计算误差
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_mae = np.mean(np.abs(y_train - y_pred_train))
        test_mae = np.mean(np.abs(y_test - y_pred_test))
        test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测性能
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_batch(
        self,
        states: List[NetworkState],
        feature_window: int = 10
    ) -> List[Dict[str, float]]:
        """
        批量预测（需要历史状态）
        
        Args:
            states: 网络状态列表
            feature_window: 特征窗口大小
            
        Returns:
            预测结果列表
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        predictions = []
        
        # 构建特征矩阵
        X = []
        for i in range(feature_window, len(states)):
            features = []
            for j in range(i - feature_window, i):
                state = states[j]
                features.extend([
                    state.arrival_rate,
                    state.service_rate,
                    state.utilization,
                    state.queue_length
                ])
            X.append(features)
        
        if len(X) == 0:
            # 如果数据不足，返回空预测
            return [{'delay': 0.0} for _ in states]
        
        X = np.array(X)
        delays_pred = self.predict(X)
        
        # 为前feature_window个状态使用实际值或简单预测
        for i in range(feature_window):
            predictions.append({
                'delay': states[i].delay,
                'utilization': states[i].utilization,
                'queue_length': states[i].queue_length
            })
        
        # 添加预测结果
        for delay in delays_pred:
            predictions.append({
                'delay': delay,
                'utilization': 0.0,  # MLP只预测延迟
                'queue_length': 0.0
            })
        
        return predictions
    
    def evaluate_prediction(
        self,
        predictions: List[Dict[str, float]],
        actual_states: List[NetworkState]
    ) -> Dict[str, float]:
        """
        评估预测准确性
        
        Args:
            predictions: 预测结果列表
            actual_states: 实际网络状态列表
            
        Returns:
            评估指标字典
        """
        if len(predictions) != len(actual_states):
            raise ValueError("预测结果和实际状态数量不匹配")
        
        delays_pred = np.array([p['delay'] for p in predictions])
        delays_actual = np.array([s.delay for s in actual_states])
        
        # 过滤无效值
        valid_mask = np.isfinite(delays_pred) & np.isfinite(delays_actual)
        delays_pred = delays_pred[valid_mask]
        delays_actual = delays_actual[valid_mask]
        
        if len(delays_pred) == 0:
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf')
            }
        
        # 计算误差指标
        mae = np.mean(np.abs(delays_pred - delays_actual))
        mse = np.mean((delays_pred - delays_actual) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((delays_actual - delays_pred) / delays_actual)) * 100
            mape = mape if not np.isnan(mape) else float('inf')
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }

