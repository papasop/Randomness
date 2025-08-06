# 📌 λ_k(x) 结构压缩 vs 实际运行时间 T(x) 实验：结构矩阵 vs 随机矩阵

import numpy as np
import time
import json
import zlib
import lzma
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from math import log2

# λ_k 计算
def compute_lambda_k(matrix_A, vector_b, method='zlib'):
    data = {'A': matrix_A.tolist(), 'b': vector_b.tolist()}
    raw_bytes = json.dumps(data).encode('utf-8')
    raw_len = len(raw_bytes)
    
    if method == 'zlib':
        compressed = zlib.compress(raw_bytes)
    elif method == 'lzma':
        compressed = lzma.compress(raw_bytes)
    else:
        raise ValueError("Unsupported compression method.")
    
    compressed_len = len(compressed)
    return compressed_len / raw_len

# 结构化矩阵（Toeplitz + 对角结构）
def generate_structured_matrix(n):
    row = np.arange(n)
    toeplitz_matrix = np.abs(np.subtract.outer(row, row))
    return toeplitz_matrix / np.max(toeplitz_matrix)

# 主实验函数
def run_experiment(n_list):
    results = []

    for n in n_list:
        print(f"\n=== Matrix size: n = {n} ===")

        for matrix_type in ['random', 'structured']:
            if matrix_type == 'random':
                A = np.random.rand(n, n)
                b = np.random.rand(n)
            else:
                A = generate_structured_matrix(n)
                b = np.linspace(0, 1, n)

            λ_zlib = compute_lambda_k(A, b, method='zlib')
            λ_lzma = compute_lambda_k(A, b, method='lzma')

            start = time.time()
            _ = np.linalg.solve(A + 1e-3*np.eye(n), b)
            end = time.time()
            T = end - start
            logT = log2(T + 1e-9)
            omega = n**2 * 2**λ_zlib

            results.append({
                'n': n,
                'type': matrix_type,
                'λ_zlib': λ_zlib,
                'λ_lzma': λ_lzma,
                'T': T,
                'log2T': logT,
                'Ω(n²·2^λk)': omega
            })

    return pd.DataFrame(results)

# ✅ 执行实验
n_list = [100, 200, 400, 800]
df = run_experiment(n_list)

# ✅ 展示结果表格
print("\n实验数据（前几行）：")
print(df.head())

# ✅ 可视化 λ_k 与 log₂T 的关系
plt.figure(figsize=(8, 5))
for matrix_type in ['random', 'structured']:
    sub = df[df['type'] == matrix_type]
    plt.plot(sub['λ_zlib'], sub['log2T'], 'o-', label=f'{matrix_type}')

plt.xlabel("λ_k (zlib)")
plt.ylabel("log₂ T(x)")
plt.title("结构 vs 随机矩阵：λ_k 与运行时间对比")
plt.grid(True)
plt.legend()
plt.show()

# ✅ 拟合回归 log₂T ≈ α·λ_k + β
for matrix_type in ['random', 'structured']:
    sub = df[df['type'] == matrix_type]
    X = sub[['λ_zlib']].values
    y = sub['log2T'].values
    model = LinearRegression().fit(X, y)
    α = model.coef_[0]
    β = model.intercept_
    print(f"{matrix_type}: log₂T ≈ {α:.2f}·λ_k + {β:.2f}")
