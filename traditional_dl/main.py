#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iris_pytorch.py

使用 PyTorch 在 GPU（若可用）上训练 Iris 鸢尾花数据集的多分类模型示例。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(test_size=0.2, random_state=42):
    # 1. 载入原始数据
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 3. 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 4. 转为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    return X_train, X_test, y_train, y_test

class IrisDataset(Dataset):
    """自定义 Dataset，用于封装特征和标签张量。"""
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class IrisNet(nn.Module):
    """一个简单的三层全连接网络：4 → 16 → 16 → 3"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)

def train_and_evaluate(
        model, device, train_loader, test_loader,
        criterion, optimizer, num_epochs=50
):
    """执行训练与评估循环，并输出训练损失与测试集准确率。"""
    for epoch in range(1, num_epochs + 1):
        # ——— 训练阶段 ———
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ——— 验证阶段 ———
        model.eval()
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()

        test_acc = correct / len(test_loader.dataset)

        # 每 10 个 epoch 或第 1 个 epoch 打印一次
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

def main():
    # 检测设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据准备
    X_train, X_test, y_train, y_test = prepare_data()

    # 构建 Dataset 与 DataLoader
    train_ds = IrisDataset(X_train, y_train)
    test_ds  = IrisDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

    # 模型、损失函数、优化器
    model = IrisNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练与评估
    train_and_evaluate(
        model, device,
        train_loader, test_loader,
        criterion, optimizer,
        num_epochs=100
    )

    # 保存模型
    torch.save(model.state_dict(), "iris_model.pth")
    print("训练完成，模型已保存为 iris_model.pth")

if __name__ == "__main__":
    main()
