import zlib
import numpy as np
from sklearn.linear_model import LinearRegression
import string
import random

# 压缩长度函数 (C(x) 的代理)
def compress_length(data):
    if isinstance(data, np.ndarray):
        data = data.tobytes()  # 数值数组转字节
    return len(zlib.compress(data))

# 生成结构化数据: y = 2x + 1 + 小噪声
def generate_structured_data(n=100):
    x = np.linspace(0, 10, n).reshape(-1, 1)
    y = 2 * x.flatten() + 1 + np.random.normal(0, 0.1, n)  # 线性 + 小噪声
    return x, y

# 生成随机数据: 纯正态噪声
def generate_random_data(n=100):
    return np.random.normal(0, 1, n)

# 计算 lambda_k
def calculate_lambda_k(x, y):
    # 拟合线性模型
    model = LinearRegression()
    model.fit(x, y.reshape(-1, 1))
    y_hat = model.predict(x).flatten()
    residual = y - y_hat  # 残差 delta(x)
    
    C_y = compress_length(y)
    C_residual = compress_length(residual)
    lambda_k = C_residual / C_y if C_y > 0 else float('inf')
    return lambda_k, C_y, C_residual

# 字符串基线测试
def string_baseline_test(n=100):
    # 结构化字符串: 重复 "abc"
    structured_str = "abc" * n
    # 随机字符串: 随机字母
    random_str = ''.join(random.choice(string.ascii_lowercase) for _ in range(3*n))
    
    C_structured = compress_length(structured_str.encode())
    C_random = compress_length(random_str.encode())
    return C_structured, C_random

# 测试
np.random.seed(42)  # 固定种子以确保可重复性
n = 100

# 测试 1: 结构化数据
x, y_structured = generate_structured_data(n)
lambda_k_structured, C_y_structured, C_residual_structured = calculate_lambda_k(x, y_structured)
print("Structured Data:")
print(f"C(y) = {C_y_structured}, C(delta(y)) = {C_residual_structured}, lambda_k = {lambda_k_structured:.4f}")

# 测试 2: 随机数据
y_random = generate_random_data(n)
x_random = np.linspace(0, 10, n).reshape(-1, 1)  # 仍提供 x 用于拟合
lambda_k_random, C_y_random, C_residual_random = calculate_lambda_k(x_random, y_random)
print("\nRandom Data:")
print(f"C(y) = {C_y_random}, C(delta(y)) = {C_residual_random}, lambda_k = {lambda_k_random:.4f}")

# 测试 3: 字符串基线
C_structured_str, C_random_str = string_baseline_test(n)
print("\nString Baseline:")
print(f"Structured string compression length: {C_structured_str}")
print(f"Random string compression length: {C_random_str}")