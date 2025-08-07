import numpy as np
import zlib
import bz2
from mpmath import mp
import matplotlib.pyplot as plt
from sympy import primerange

# 设置精度
mp.dps = 100000 + 5

# 生成序列
def get_rational(n):
    return '3' * n  # 有理数：0.333...
def get_primes(n):
    primes = [str(p) for p in primerange(2, 10**6)]
    return ''.join(primes)[:n]  # 素数序列：235711...
def get_transcendental(n, number='pi'):
    if number == 'pi':
        return str(mp.pi)[2:2+n]
    elif number == 'e':
        return str(mp.e)[2:2+n]
    else:
        raise ValueError("Unsupported number")
def get_random(n):
    return ''.join(np.random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], n))

# 压缩函数
def compress_length(data_str, method='zlib'):
    data_bytes = data_str.encode('utf-8')
    if method == 'zlib':
        return len(zlib.compress(data_bytes))
    elif method == 'bz2':
        return len(bz2.compress(data_bytes))
    else:
        raise ValueError("Unsupported compression method")

# 测试 λ_k
lengths = [100, 500, 1000, 5000, 10000, 50000, 100000]
results_rational = []
results_primes = []
results_pi = []
results_e = []
results_random = []

for n in lengths:
    # 有理数 (1/3)
    rational_digits = get_rational(n)
    rational_approx = '3' * n
    rational_residual = ''.join(chr(ord(a) ^ ord(b)) for a, b in zip(rational_digits, rational_approx))
    C_rational_zlib = compress_length(rational_digits, 'zlib')
    C_rational_res_zlib = compress_length(rational_residual, 'zlib')
    C_rational_bz2 = compress_length(rational_digits, 'bz2')
    C_rational_res_bz2 = compress_length(rational_residual, 'bz2')
    lambda_k_rational_zlib = C_rational_res_zlib / C_rational_zlib if C_rational_zlib != 0 else float('inf')
    lambda_k_rational_bz2 = C_rational_res_bz2 / C_rational_bz2 if C_rational_bz2 != 0 else float('inf')
    results_rational.append((n, lambda_k_rational_zlib, lambda_k_rational_bz2, C_rational_zlib, C_rational_res_zlib))

    # 素数序列
    primes_digits = get_primes(n)
    primes_approx = (primes_digits[:min(100, n)] * (n // min(100, n) + 1))[:n]  # 优化逼近
    primes_residual = ''.join(chr(ord(a) ^ ord(b)) for a, b in zip(primes_digits, primes_approx))
    C_primes_zlib = compress_length(primes_digits, 'zlib')
    C_primes_res_zlib = compress_length(primes_residual, 'zlib')
    C_primes_bz2 = compress_length(primes_digits, 'bz2')
    C_primes_res_bz2 = compress_length(primes_residual, 'bz2')
    lambda_k_primes_zlib = C_primes_res_zlib / C_primes_zlib if C_primes_zlib != 0 else float('inf')
    lambda_k_primes_bz2 = C_primes_res_bz2 / C_primes_bz2 if C_primes_bz2 != 0 else float('inf')
    results_primes.append((n, lambda_k_primes_zlib, lambda_k_primes_bz2, C_primes_zlib, C_primes_res_zlib))

    # 超越数 (π)
    pi_digits = get_transcendental(n, 'pi')
    pi_approx = pi_digits[:3*n//4] + (pi_digits[:min(100, n)] * ((n - 3*n//4) // min(100, n) + 1))[:n-3*n//4] if n > 100 else pi_digits[:min(100, n)]
    pi_residual = ''.join(chr(ord(a) ^ ord(b)) for a, b in zip(pi_digits, pi_approx))
    C_pi_zlib = compress_length(pi_digits, 'zlib')
    C_pi_res_zlib = compress_length(pi_residual, 'zlib')
    C_pi_bz2 = compress_length(pi_digits, 'bz2')
    C_pi_res_bz2 = compress_length(pi_residual, 'bz2')
    lambda_k_pi_zlib = C_pi_res_zlib / C_pi_zlib if C_pi_zlib != 0 else float('inf')
    lambda_k_pi_bz2 = C_pi_res_bz2 / C_pi_bz2 if C_pi_bz2 != 0 else float('inf')
    C_pi_n_zlib = C_pi_zlib / n
    results_pi.append((n, lambda_k_pi_zlib, lambda_k_pi_bz2, C_pi_zlib, C_pi_res_zlib, C_pi_n_zlib))

    # 超越数 (e)
    e_digits = get_transcendental(n, 'e')
    e_approx = e_digits[:3*n//4] + (e_digits[:min(100, n)] * ((n - 3*n//4) // min(100, n) + 1))[:n-3*n//4] if n > 100 else e_digits[:min(100, n)]
    e_residual = ''.join(chr(ord(a) ^ ord(b)) for a, b in zip(e_digits, e_approx))
    C_e_zlib = compress_length(e_digits, 'zlib')
    C_e_res_zlib = compress_length(e_residual, 'zlib')
    C_e_bz2 = compress_length(e_digits, 'bz2')
    C_e_res_bz2 = compress_length(e_residual, 'bz2')
    lambda_k_e_zlib = C_e_res_zlib / C_e_zlib if C_e_zlib != 0 else float('inf')
    lambda_k_e_bz2 = C_e_res_bz2 / C_e_bz2 if C_e_bz2 != 0 else float('inf')
    C_e_n_zlib = C_e_zlib / n
    results_e.append((n, lambda_k_e_zlib, lambda_k_e_bz2, C_e_zlib, C_e_res_zlib, C_e_n_zlib))

    # 真随机序列
    random_digits = get_random(n)
    random_approx = (random_digits[:min(100, n)] * (n // min(100, n) + 1))[:n]
    random_residual = ''.join(chr(ord(a) ^ ord(b)) for a, b in zip(random_digits, random_approx))
    C_random_zlib = compress_length(random_digits, 'zlib')
    C_random_res_zlib = compress_length(random_residual, 'zlib')
    C_random_bz2 = compress_length(random_digits, 'bz2')
    C_random_res_bz2 = compress_length(random_residual, 'bz2')
    lambda_k_random_zlib = C_random_res_zlib / C_random_zlib if C_random_zlib != 0 else float('inf')
    lambda_k_random_bz2 = C_random_res_bz2 / C_random_bz2 if C_random_bz2 != 0 else float('inf')
    results_random.append((n, lambda_k_random_zlib, lambda_k_random_bz2, C_random_zlib, C_random_res_zlib))

# 输出结果
print("有理数 (1/3):")
for n, lambda_k_zlib, lambda_k_bz2, C_rational, C_rational_res in results_rational:
    print(f"n={n}: zlib λ_k={lambda_k_zlib:.3f}, bz2 λ_k={lambda_k_bz2:.3f}, C(x)={C_rational}, C(δ)={C_rational_res}")

print("\n素数序列:")
for n, lambda_k_zlib, lambda_k_bz2, C_primes, C_primes_res in results_primes:
    print(f"n={n}: zlib λ_k={lambda_k_zlib:.3f}, bz2 λ_k={lambda_k_bz2:.3f}, C(x)={C_primes}, C(δ)={C_primes_res}")

print("\nπ:")
for n, lambda_k_zlib, lambda_k_bz2, C_pi, C_pi_res, C_pi_n in results_pi:
    print(f"n={n}: zlib λ_k={lambda_k_zlib:.3f}, bz2 λ_k={lambda_k_bz2:.3f}, C(π)={C_pi}, C(δ)={C_pi_res}, C(π)/n={C_pi_n:.3f}")

print("\ne:")
for n, lambda_k_zlib, lambda_k_bz2, C_e, C_e_res, C_e_n in results_e:
    print(f"n={n}: zlib λ_k={lambda_k_zlib:.3f}, bz2 λ_k={lambda_k_bz2:.3f}, C(e)={C_e}, C(δ)={C_e_res}, C(e)/n={C_e_n:.3f}")

print("\n真随机序列:")
for n, lambda_k_zlib, lambda_k_bz2, C_random, C_random_res in results_random:
    print(f"n={n}: zlib λ_k={lambda_k_zlib:.3f}, bz2 λ_k={lambda_k_bz2:.3f}, C(x)={C_random}, C(δ)={C_random_res}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(lengths, [r[1] for r in results_rational], marker='o', label='Rational (zlib λ_k)')
plt.plot(lengths, [r[1] for r in results_primes], marker='s', label='Primes (zlib λ_k)')
plt.plot(lengths, [r[1] for r in results_pi], marker='^', label='π (zlib λ_k)')
plt.plot(lengths, [r[1] for r in results_e], marker='d', label='e (zlib λ_k)')
plt.plot(lengths, [r[1] for r in results_random], marker='x', label='Random (zlib λ_k)')
plt.xscale('log')
plt.xlabel('Sequence Length (n)')
plt.ylabel('λ_k')
plt.title('Convergence of λ_k for Different Sequence Types')
plt.legend()
plt.grid(True)
plt.savefig('sequence_lambda_k_comparison.png')
plt.show()