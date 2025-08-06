import numpy as np
import zlib
import bz2
import matplotlib.pyplot as plt
from mpmath import mp
import sys

# 提高整数字符串转换限制
sys.set_int_max_str_digits(1000000)  # 允许百万位整数转换

# 定义压缩函数
def compressed_size(data: bytes, method='zlib') -> int:
    if method == 'zlib':
        return len(zlib.compress(data))
    elif method == 'bz2':
        return len(bz2.compress(data))
    else:
        raise NotImplementedError("Only zlib and bz2 supported")

# 生成 Fibonacci 序列
def fibonacci_string(n):
    fib = [0, 1]
    result = []
    total_len = 0
    for i in range(2, n):  # 限制生成，避免过大
        fib.append(fib[i-1] + fib[i-2])
        str_fib = str(fib[i])
        result.append(str_fib)
        total_len += len(str_fib)
        if total_len >= n:
            break
    return ''.join(result)[:n]  # 截取前 n 位

# 测试序列长度
lengths = [100, 500, 1000, 5000, 10000, 50000, 100000]
lambda_k_zlib_pi = []
lambda_k_bz2_pi = []
compression_ratio_zlib_pi = []
compression_ratio_bz2_pi = []
lambda_k_zlib_fib = []
lambda_k_bz2_fib = []
lambda_k_zlib_rand = []
lambda_k_bz2_rand = []
C_pi_zlib_list = []
C_res_zlib_list = []
C_pi_bz2_list = []
C_res_bz2_list = []

for n in lengths:
    # 生成 π 的前 n 位小数
    mp.dps = n + 5
    pi_str = str(mp.pi)[2:n+2]  # 去掉 "3."
    pi_bytes = pi_str.encode('utf-8')

    # 优化逼近模型：前 3n/4 位取真实 π，剩余重复前100位
    prefix = pi_str[:min(100, n)]
    if n <= 100:
        approx_str = prefix
    else:
        approx_str = pi_str[:3*n//4] + (prefix * ((n - 3*n//4) // len(prefix) + 1))[:n-3*n//4]
    approx_bytes = approx_str.encode('utf-8')

    # 残差：逐字节异或
    residual_bytes = bytes([a ^ b for a, b in zip(pi_bytes, approx_bytes)])

    # 生成 Fibonacci 和随机序列
    fib_str = fibonacci_string(n)
    fib_bytes = fib_str.encode('utf-8')
    rand_str = ''.join(np.random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], n))
    rand_bytes = rand_str.encode('utf-8')

    # Fibonacci 和随机序列的逼近（简单重复前100位）
    fib_approx = (fib_str[:min(100, n)] * (n // len(fib_str[:min(100, n)]) + 1))[:n]
    fib_approx_bytes = fib_approx.encode('utf-8')
    rand_approx = (rand_str[:min(100, n)] * (n // len(rand_str[:min(100, n)]) + 1))[:n]
    rand_approx_bytes = rand_approx.encode('utf-8')

    # Fibonacci 和随机序列残差
    fib_residual = bytes([a ^ b for a, b in zip(fib_bytes, fib_approx_bytes)])
    rand_residual = bytes([a ^ b for a, b in zip(rand_bytes, rand_approx_bytes)])

    # 计算压缩大小
    C_pi_zlib = compressed_size(pi_bytes, 'zlib')
    C_res_zlib = compressed_size(residual_bytes, 'zlib')
    C_pi_bz2 = compressed_size(pi_bytes, 'bz2')
    C_res_bz2 = compressed_size(residual_bytes, 'bz2')
    C_fib_zlib = compressed_size(fib_bytes, 'zlib')
    C_fib_res_zlib = compressed_size(fib_residual, 'zlib')
    C_rand_zlib = compressed_size(rand_bytes, 'zlib')
    C_rand_res_zlib = compressed_size(rand_residual, 'zlib')
    C_fib_bz2 = compressed_size(fib_bytes, 'bz2')
    C_fib_res_bz2 = compressed_size(fib_residual, 'bz2')
    C_rand_bz2 = compressed_size(rand_bytes, 'bz2')
    C_rand_res_bz2 = compressed_size(rand_residual, 'bz2')

    # 存储结果
    lambda_k_zlib_pi.append(C_res_zlib / C_pi_zlib)
    lambda_k_bz2_pi.append(C_res_bz2 / C_pi_bz2)
    compression_ratio_zlib_pi.append(C_pi_zlib / n)
    compression_ratio_bz2_pi.append(C_pi_bz2 / n)
    lambda_k_zlib_fib.append(C_fib_res_zlib / C_fib_zlib)
    lambda_k_bz2_fib.append(C_fib_res_bz2 / C_fib_bz2)
    lambda_k_zlib_rand.append(C_rand_res_zlib / C_rand_zlib)
    lambda_k_bz2_rand.append(C_rand_res_bz2 / C_rand_zlib)
    C_pi_zlib_list.append(C_pi_zlib)
    C_res_zlib_list.append(C_res_zlib)
    C_pi_bz2_list.append(C_pi_bz2)
    C_res_bz2_list.append(C_res_bz2)

# 输出结果
for i, n in enumerate(lengths):
    print(f"n={n}:")
    print(f"  π (zlib): λ_k={lambda_k_zlib_pi[i]:.3f}, C(π)={C_pi_zlib_list[i]}, C(δ)={C_res_zlib_list[i]}, C(π)/n={compression_ratio_zlib_pi[i]:.3f}")
    print(f"  π (bz2): λ_k={lambda_k_bz2_pi[i]:.3f}, C(π)={C_pi_bz2_list[i]}, C(δ)={C_res_bz2_list[i]}, C(π)/n={compression_ratio_bz2_pi[i]:.3f}")
    print(f"  Fibonacci (zlib): λ_k={lambda_k_zlib_fib[i]:.3f}")
    print(f"  Fibonacci (bz2): λ_k={lambda_k_bz2_fib[i]:.3f}")
    print(f"  Random (zlib): λ_k={lambda_k_zlib_rand[i]:.3f}")
    print(f"  Random (bz2): λ_k={lambda_k_bz2_rand[i]:.3f}")

# 可视化
plt.figure(figsize=(12, 8))
plt.plot(lengths, lambda_k_zlib_pi, marker='o', label='π (zlib λ_k)')
plt.plot(lengths, lambda_k_bz2_pi, marker='s', label='π (bz2 λ_k)')
plt.plot(lengths, lambda_k_zlib_fib, marker='o', linestyle='--', label='Fibonacci (zlib λ_k)')
plt.plot(lengths, lambda_k_bz2_fib, marker='s', linestyle='--', label='Fibonacci (bz2 λ_k)')
plt.plot(lengths, lambda_k_zlib_rand, marker='o', linestyle=':', label='Random (zlib λ_k)')
plt.plot(lengths, lambda_k_bz2_rand, marker='s', linestyle=':', label='Random (bz2 λ_k)')
plt.xscale('log')
plt.xlabel('序列长度 (n)')
plt.ylabel('λ_k')
plt.title('π、Fibonacci 和随机序列的 λ_k 收敛趋势')
plt.legend()
plt.grid(True)
plt.savefig('pi_lambda_k_comparison.png')
plt.show()