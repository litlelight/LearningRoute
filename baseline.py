import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import shap


class DataPreprocessor:
    def __init__(self):
        # 按照SRL理论将特征分为三个阶段
        self.forethought_features = ['Motivation_Level', 'Parental_Education_Level',
                                     'Family_Income', 'School_Type', 'Access_to_Resources']
        self.performance_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours',
                                     'Tutoring_Sessions', 'Physical_Activity']
        self.reflection_features = ['Previous_Scores', 'Teacher_Quality',
                                    'Peer_Influence', 'Learning_Disabilities']

        self.numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours',
                                   'Previous_Scores', 'Tutoring_Sessions',
                                   'Physical_Activity', 'Distance_from_Home']
        self.categorical_features = ['Parental_Involvement', 'Access_to_Resources',
                                     'Extracurricular_Activities', 'Motivation_Level',
                                     'Internet_Access', 'Family_Income', 'Teacher_Quality',
                                     'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                                     'Parental_Education_Level', 'Gender']
        self.target = 'Exam_Score'
        self.scalers = {}
        self.encoders = {}

    def preprocess(self, data):
        df = data.copy()

        # 数值特征标准化
        for feature in self.numerical_features:
            self.scalers[feature] = StandardScaler()
            df[feature] = self.scalers[feature].fit_transform(df[[feature]])

        # 类别特征编码
        for feature in self.categorical_features:
            self.encoders[feature] = LabelEncoder()
            df[feature] = self.encoders[feature].fit_transform(df[feature])

        # 按SRL阶段分组特征
        X_forethought = df[self.forethought_features].values
        X_performance = df[self.performance_features].values
        X_reflection = df[self.reflection_features].values

        # 合并所有特征
        X = df[self.numerical_features + self.categorical_features].values
        y = df[self.target].values

        return X, y, (X_forethought, X_performance, X_reflection)

    def create_graph(self, X):
        N = X.shape[0]
        edge_index = []
        edge_weights = []

        # 使用特征相似度构建动态图
        for i in range(N):
            distances = np.linalg.norm(X - X[i], axis=1)
            nearest_neighbors = np.argsort(distances)[1:6]  # 取最近的5个邻居

            for j in nearest_neighbors:
                edge_index.append([i, j])
                # 计算基于特征相似度的边权重
                similarity = 1 / (1 + distances[j])
                edge_weights.append([similarity])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        x = torch.tensor(X, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_weights)


class SRLAttention(nn.Module):
    def __init__(self, input_dim, num_heads=3):  # 3个head对应SRL三个阶段
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # 为每个SRL阶段创建独立的attention head
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        B, N, C = x.size()

        # 分离query, key, value到不同head
        q = self.query(x).view(B, N, self.num_heads, self.head_dim)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim)

        # 计算attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # 应用attention
        out = torch.matmul(attn_weights, v)
        out = out.reshape(B, N, -1)

        return self.output_layer(out), attn_weights


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, h_n


class DynamicWeightModule(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

    def forward(self):
        weights = F.softmax(self.task_weights, dim=0)
        return weights


class EnhancedGNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        # SRL阶段特定的GNN层
        self.forethought_gnn = GATConv(input_dim, hidden_dim, heads=4)
        self.performance_gnn = GATConv(input_dim, hidden_dim, heads=4)
        self.reflection_gnn = GATConv(input_dim, hidden_dim, heads=4)

        # SRL attention机制
        self.srl_attention = SRLAttention(hidden_dim * 4)

        # 时序编码器
        self.temporal_encoder = TemporalEncoder(hidden_dim * 4, hidden_dim)

        # 动态任务权重
        self.dynamic_weights = DynamicWeightModule()

        # 预测分数的层
        self.score_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

        # 生成建议的层
        self.recommendation_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 5)  # 5种建议类别
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch_size = x.size(0)

        # 1. SRL阶段特征提取
        x_forethought = F.relu(self.forethought_gnn(x, edge_index))
        x_performance = F.relu(self.performance_gnn(x, edge_index))
        x_reflection = F.relu(self.reflection_gnn(x, edge_index))

        # 2. 特征融合与attention
        combined_features = torch.cat([x_forethought, x_performance, x_reflection], dim=-1)
        attended_features, attention_weights = self.srl_attention(combined_features.unsqueeze(0))

        # 3. 时序建模
        temporal_features, _ = self.temporal_encoder(attended_features)

        # 4. 特征整合
        final_representation = torch.cat([
            attended_features.squeeze(0),
            temporal_features.squeeze(0)
        ], dim=-1)

        # 5. 多任务预测
        scores = self.score_predictor(final_representation)
        recommendations = self.recommendation_generator(final_representation)

        # 6. 获取动态任务权重
        task_weights = self.dynamic_weights()

        return scores, recommendations, task_weights, attention_weights


def train_model(model, graph_data, targets, recommendation_targets=None,
                num_epochs=100, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    graph_data = graph_data.to(device)
    targets = targets.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    score_criterion = nn.MSELoss()
    recommendation_criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience = 10
    counter = 0

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 前向传播
        scores, recommendations, task_weights, _ = model(graph_data)

        # 计算损失
        score_loss = score_criterion(scores.squeeze(), targets)
        if recommendation_targets is not None:
            recommendation_loss = recommendation_criterion(recommendations, recommendation_targets)
        else:
            recommendation_loss = torch.tensor(0.0).to(device)

        # 使用动态权重组合损失
        total_loss = task_weights[0] * score_loss + task_weights[1] * recommendation_loss

        # 反向传播
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # 早停
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Total Loss: {total_loss.item():.4f}')
            print(f'Score Loss: {score_loss.item():.4f}')
            print(f'Task Weights: {task_weights.detach().cpu().numpy()}')
            print('-' * 50)


def main():
    # 1. 加载数据
    print("Loading data...")
    data = pd.read_csv('data01.csv')

    # 2. 数据预处理
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y, srl_features = preprocessor.preprocess(data)

    # 3. 创建图数据
    print("Creating graph data...")
    graph_data = preprocessor.create_graph(X)
    targets = torch.tensor(y, dtype=torch.float)

    # 4. 创建模型
    print("Initializing model...")
    model = EnhancedGNNModel(input_dim=X.shape[1])

    # 5. 训练模型
    print("Training model...")
    train_model(model, graph_data, targets)

    print("Training completed!")


if __name__ == "__main__":
    main()