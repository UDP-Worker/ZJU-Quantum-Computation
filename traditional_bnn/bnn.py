#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iris_bnn_enhanced.py
改进版二值神经网络 (BNN) 解决 Iris 分类问题
核心改动
1. 首层与末层保留浮点权重以减少信息丢失
2. SignSTE 采用分段线性近似, 改善梯度流
3. 隐藏层宽度扩展至 32 并增加深度, 提升容量
4. 引入余弦退火学习率调度与轻度权重衰减, 稳定训练
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

# ---------- 0. 可复现设置 ----------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ---------- 1. 数据准备 ----------
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

class IrisDS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(IrisDS(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(IrisDS(X_test, y_test), batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. 二值化算子 ----------
class SignSTE(torch.autograd.Function):
    """Sign 函数的直通估计 (Bi-Real Net 分段线性近似)"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # |x|<=1 区域梯度为 1, 其余为 0
        mask = (x.abs() <= 1).float()
        return grad_output * mask

def binarize(x):
    return SignSTE.apply(x)

# ---------- 3. 二值线性层 ----------
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        alpha = self.weight.abs().mean(dim=1, keepdim=True)
        w_bin = binarize(self.weight) * alpha
        return nn.functional.linear(x, w_bin, self.bias)

# ---------- 4. BNN 网络结构 ----------
class IrisBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(4, 32, bias=False)  # 浮点首层
        self.bn_in = nn.BatchNorm1d(32)

        self.b1 = BinaryLinear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.b2 = BinaryLinear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc_out = nn.Linear(32, 3, bias=False)  # 浮点末层

    def forward(self, x):
        x = binarize(self.bn_in(self.fc_in(x)))
        x = binarize(self.bn1(self.b1(x)))
        x = binarize(self.bn2(self.b2(x)))
        return self.fc_out(x)

model = IrisBNN().to(device)

# ---------- 5. 损失、优化器与调度 ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# ---------- 6. 训练 & 评估 ----------

def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
    return correct / len(loader.dataset)

num_epochs = 200
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    scheduler.step()

    if epoch == 1 or epoch % 10 == 0:
        train_loss = total_loss / len(train_loader.dataset)
        test_acc = evaluate(test_loader)
        print(f"Epoch {epoch:3d} | TrainLoss {train_loss:.4f} | TestAcc {test_acc:.4f}")

print("\nFinal test accuracy:", evaluate(test_loader))
