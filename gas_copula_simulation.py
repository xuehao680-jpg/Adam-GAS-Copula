"""
GAS-Copula vs GAS 模型对比分析

本脚本演示：
1. 动态相关系数数据模拟
2. GAS-Copula 模型拟合
3. 标准 GAS 模型拟合
4. 性能对比分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import ndtri
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 数据模拟模块
# ============================================================================

def simulate_dynamic_correlation(
    n=500,
    pattern='sinusoidal',
    seed=None,
    **kwargs
):
    """
    模拟动态相关系数

    Parameters
    ----------
    n : int
        样本量
    pattern : str
        相关系数模式：
        - 'sinusoidal': 正弦波动
        - 'regime': 区制转换
        - 'gradual': 渐变
        - 'volatile': 高波动
        - 'mixed': 混合模式
    seed : int
        随机种子

    Returns
    -------
    rho_true : array
        真实相关系数路径
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n)

    if pattern == 'sinusoidal':
        # 周期性波动
        period = kwargs.get('period', 100)
        rho_true = 0.3 + 0.4 * np.sin(2 * np.pi * t / period)

    elif pattern == 'regime':
        # 区制转换
        rho_true = np.zeros(n)
        regime_length = n // 4
        rho_true[:regime_length] = 0.2
        rho_true[regime_length:2*regime_length] = 0.7
        rho_true[2*regime_length:3*regime_length] = 0.0
        rho_true[3*regime_length:] = 0.5
        # 添加一些噪声
        rho_true += 0.05 * np.random.randn(n)

    elif pattern == 'gradual':
        # 渐变
        rho_true = 0.1 + 0.6 * t / n

    elif pattern == 'volatile':
        # 高波动
        rho_true = 0.3 + 0.3 * np.cumsum(np.random.randn(n)) / np.sqrt(n)
        rho_true = np.clip(rho_true, -0.8, 0.8)

    elif pattern == 'mixed':
        # 混合模式：趋势 + 周期 + 随机
        trend = 0.1 + 0.3 * t / n
        seasonal = 0.2 * np.sin(2 * np.pi * t / 50)
        noise = 0.1 * np.cumsum(np.random.randn(n)) / np.sqrt(n)
        rho_true = trend + seasonal + noise

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # 确保相关系数在有效范围内
    rho_true = np.clip(rho_true, -0.95, 0.95)

    return rho_true


def simulate_copula_data(
    n=500,
    rho_true=None,
    marginal_x='normal',
    marginal_y='normal',
    seed=None
):
    """
    基于 Copula 模拟双变量数据

    Parameters
    ----------
    n : int
        样本量
    rho_true : array
        真实相关系数路径
    marginal_x, marginal_y : str
        边缘分布类型: 'normal', 't', 'exponential', 'uniform'
    seed : int
        随机种子

    Returns
    -------
    x, y : array
        模拟的数据
    """
    if seed is not None:
        np.random.seed(seed)

    if rho_true is None:
        rho_true = simulate_dynamic_correlation(n)

    # 生成二元正态数据
    x = np.zeros(n)
    y = np.zeros(n)

    for t in range(n):
        rho_t = np.clip(rho_true[t], -0.999, 0.999)
        # Cholesky 分解
        L = np.array([[1, 0],
                      [rho_t, np.sqrt(1 - rho_t**2)]])
        z = np.random.randn(2)
        xy = L @ z
        x[t] = xy[0]
        y[t] = xy[1]

    # 变换边缘分布
    x = _transform_marginal(x, marginal_x)
    y = _transform_marginal(y, marginal_y)

    return x, y, rho_true


def _transform_marginal(data, marginal_type):
    """变换边缘分布"""
    if marginal_type == 'normal':
        return data
    elif marginal_type == 't':
        # t 分布（自由度=5）
        z = stats.norm.cdf(data)
        return stats.t.ppf(z, df=5)
    elif marginal_type == 'exponential':
        # 指数分布
        z = stats.norm.cdf(data)
        return stats.expon.ppf(z)
    elif marginal_type == 'uniform':
        # 均匀分布
        return stats.norm.cdf(data)
    else:
        return data


# ============================================================================
# 2. 简化版 GAS-Copula 实现（用于对比）
# ============================================================================

class SimpleGASCopula:
    """
    简化版 GAS-Copula 模型
    用于快速拟合和对比
    """

    def __init__(self, ar=1, sc=1):
        self.ar = ar
        self.sc = sc
        self.params = None
        self.rho_path = None
        self._fitted = False

    def _transform_to_unconstrained(self, rho):
        """Fisher z 变换"""
        return np.arctanh(np.clip(rho, -0.999, 0.999))

    def _transform_to_constrained(self, f):
        """逆变换"""
        return np.tanh(np.clip(f, -5, 5))

    def _gaussian_copula_loglik(self, u, v, rho):
        """高斯 Copula 对数似然"""
        rho = np.clip(rho, -0.999, 0.999)
        z1, z2 = ndtri(u), ndtri(v)
        term1 = -0.5 * np.log(1 - rho**2)
        term2 = -(rho**2 * (z1**2 + z2**2) - 2 * rho * z1 * z2) / (2 * (1 - rho**2))
        return term1 + term2

    def _gaussian_copula_score(self, u, v, rho):
        """高斯 Copula 得分函数"""
        rho = np.clip(rho, -0.999, 0.999)
        z1, z2 = ndtri(np.clip(u, 1e-10, 1-1e-10)), ndtri(np.clip(v, 1e-10, 1-1e-10))
        denom = (1 - rho**2)**2
        numerator = (z1 * z2 - rho) * (1 - rho**2) + rho * (z1**2 + z2**2 - 2 * z1 * z2 * rho)
        return numerator / denom

    def _fisher_information(self, rho):
        """Fisher 信息"""
        rho = np.clip(rho, -0.999, 0.999)
        return (1 + rho**2) / (1 - rho**2)**2

    def _gas_recursion(self, params, u, v):
        """GAS 递归"""
        n = len(u)
        omega = params[0]
        ar_coefs = params[1:1+self.ar] if self.ar > 0 else np.array([])
        sc_coefs = params[1+self.ar:] if self.sc > 0 else np.array([])

        max_lag = max(self.ar, self.sc, 1)

        f = np.zeros(n)
        theta = np.zeros(n)
        scores = np.zeros(n)

        # 初始值
        z1, z2 = ndtri(u), ndtri(v)
        init_rho = np.corrcoef(z1, z2)[0, 1]
        f[:max_lag] = self._transform_to_unconstrained(init_rho)
        theta[:max_lag] = init_rho

        total_ll = 0.0

        for t in range(max_lag, n):
            # 计算得分
            score_val = self._gaussian_copula_score(u[t-1], v[t-1], theta[t-1])
            if not np.isfinite(score_val):
                score_val = 0.0
            scores[t-1] = score_val

            # Fisher 缩放
            fisher = self._fisher_information(theta[t-1])
            if not np.isfinite(fisher) or fisher < 1e-10:
                fisher = 1.0
            scaled_score = scores[t-1] / np.sqrt(fisher)

            # GAS 更新
            ar_term = np.sum(ar_coefs * f[t-self.ar:t][::-1]) if self.ar > 0 else 0
            sc_term = np.sum(sc_coefs * scaled_score) if self.sc > 0 else 0

            f[t] = omega + ar_term + sc_term
            f[t] = np.clip(f[t], -5, 5)
            theta[t] = self._transform_to_constrained(f[t])

            # 对数似然
            ll = self._gaussian_copula_loglik(u[t], v[t], theta[t])
            if np.isfinite(ll):
                total_ll += ll

        return theta, scores, total_ll

    def fit(self, x, y, verbose=True):
        """拟合模型"""
        from scipy.optimize import minimize

        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        n = len(x)

        # 变换到均匀分布
        u = np.clip(stats.rankdata(x) / (n + 1), 1e-10, 1-1e-10)
        v = np.clip(stats.rankdata(y) / (n + 1), 1e-10, 1-1e-10)

        # 初始参数
        n_params = 1 + self.ar + self.sc
        init_params = np.zeros(n_params)
        init_params[0] = 0.0
        if self.ar > 0:
            init_params[1:1+self.ar] = 0.5 / self.ar
        if self.sc > 0:
            init_params[1+self.ar:] = 0.1 / self.sc

        def neg_loglik(params):
            _, _, ll = self._gas_recursion(params, u, v)
            return -ll

        result = minimize(
            neg_loglik,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'disp': False}
        )

        self.params = result.x
        self.rho_path, self.scores, self.ll = self._gas_recursion(self.params, u, v)
        self._fitted = True

        if verbose:
            print(f"GAS-Copula 拟合完成")
            print(f"  对数似然: {self.ll:.4f}")
            print(f"  omega: {self.params[0]:.4f}")

        return self

    def get_dynamic_correlation(self):
        return self.rho_path


class SimpleGASAdam:
    """
    简化版 GAS-Adam Copula 模型
    """

    def __init__(self, beta1=0.9, beta2=0.999, learning_rate=0.05, ar_coef=0.85):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = learning_rate
        self.A = ar_coef
        self.epsilon = 1e-8
        self.omega = 0.0
        self.rho_path = None
        self._fitted = False

    def _transform_to_unconstrained(self, rho):
        return np.arctanh(np.clip(rho, -0.999, 0.999))

    def _transform_to_constrained(self, f):
        return np.tanh(np.clip(f, -5, 5))

    def _gaussian_copula_score(self, u, v, rho):
        rho = np.clip(rho, -0.999, 0.999)
        z1, z2 = ndtri(np.clip(u, 1e-10, 1-1e-10)), ndtri(np.clip(v, 1e-10, 1-1e-10))
        denom = (1 - rho**2)**2
        numerator = (z1 * z2 - rho) * (1 - rho**2) + rho * (z1**2 + z2**2 - 2 * z1 * z2 * rho)
        return numerator / denom

    def _fisher_information(self, rho):
        rho = np.clip(rho, -0.999, 0.999)
        return (1 + rho**2) / (1 - rho**2)**2

    def fit(self, x, y, verbose=True):
        """拟合模型"""
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        n = len(x)

        # 变换到均匀分布
        u = np.clip(stats.rankdata(x) / (n + 1), 1e-10, 1-1e-10)
        v = np.clip(stats.rankdata(y) / (n + 1), 1e-10, 1-1e-10)

        # 初始化
        z1, z2 = ndtri(u), ndtri(v)
        init_rho = np.corrcoef(z1, z2)[0, 1]

        rho_path = np.zeros(n)
        f_path = np.zeros(n)

        f_path[0] = self._transform_to_unconstrained(init_rho)
        rho_path[0] = init_rho

        # Adam 状态
        m = 0.0
        v_adam = 0.0  # 改名避免与参数 v 冲突

        total_nll = 0.0

        for t in range(1, n):
            # 计算得分
            score = self._gaussian_copula_score(u[t-1], v[t-1], rho_path[t-1])

            # Fisher 缩放
            I = self._fisher_information(rho_path[t-1])
            score = score / np.sqrt(max(I, 0.01))

            if not np.isfinite(score):
                score = 0.0
            score = np.clip(score, -5, 5)

            # Adam 更新
            m = self.beta1 * m + (1 - self.beta1) * score
            v_adam = self.beta2 * v_adam + (1 - self.beta2) * score**2

            m_hat = m / (1 - self.beta1**t)
            v_hat = v_adam / (1 - self.beta2**t)

            adam_update = self.eta * m_hat / (np.sqrt(v_hat) + self.epsilon)
            f_path[t] = self.omega + self.A * f_path[t-1] + adam_update
            f_path[t] = np.clip(f_path[t], -5, 5)

            rho_path[t] = self._transform_to_constrained(f_path[t])

            # 计算损失
            ll = -0.5 * np.log(1 - rho_path[t]**2)
            ll -= (rho_path[t]**2 * (z1[t]**2 + z2[t]**2) - 2 * rho_path[t] * z1[t] * z2[t]) / (2 * (1 - rho_path[t]**2))
            total_nll -= ll

        self.rho_path = rho_path
        self._fitted = True
        self.total_nll = total_nll

        if verbose:
            print(f"GAS-Adam 拟合完成")
            print(f"  总损失 (NLL): {total_nll:.4f}")
            print(f"  参数均值: {np.mean(self.rho_path):.4f}")

        return self

    def get_dynamic_correlation(self):
        return self.rho_path


# ============================================================================
# 3. 滚动窗口相关系数（基准方法）
# ============================================================================

def rolling_correlation(x, y, window=20):
    """
    计算滚动窗口相关系数
    作为基准对比方法
    """
    n = len(x)
    rho_rolling = np.full(n, np.nan)

    for t in range(window, n):
        rho_rolling[t] = np.corrcoef(x[t-window:t], y[t-window:t])[0, 1]

    # 前面的值用第一个有效值填充
    rho_rolling[:window] = rho_rolling[window]

    return rho_rolling


# ============================================================================
# 4. DCC-GARCH 简化实现（另一个基准）
# ============================================================================

def dcc_garch_simplified(x, y, init_alpha=0.05, init_beta=0.9):
    """
    简化版 DCC-GARCH 动态相关系数

    Parameters
    ----------
    x, y : array
        数据序列
    init_alpha : float
        DCC alpha 参数
    init_beta : float
        DCC beta 参数
    """
    n = len(x)

    # 标准化数据
    x_std = (x - np.mean(x)) / np.std(x)
    y_std = (y - np.mean(y)) / np.std(y)

    # 简单的 EWMA 波动率
    lambda_ewma = 0.94

    sigma_x = np.zeros(n)
    sigma_y = np.zeros(n)
    sigma_x[0] = np.std(x)
    sigma_y[0] = np.std(y)

    for t in range(1, n):
        sigma_x[t] = np.sqrt(lambda_ewma * sigma_x[t-1]**2 + (1-lambda_ewma) * x_std[t-1]**2)
        sigma_y[t] = np.sqrt(lambda_ewma * sigma_y[t-1]**2 + (1-lambda_ewma) * y_std[t-1]**2)

    # 标准化残差
    eps_x = x_std / sigma_x
    eps_y = y_std / sigma_y

    # DCC 相关性
    alpha = init_alpha
    beta = init_beta

    rho_dcc = np.zeros(n)
    q_bar = np.mean(eps_x * eps_y)

    q = q_bar
    for t in range(1, n):
        q = (1 - alpha - beta) * q_bar + alpha * eps_x[t-1] * eps_y[t-1] + beta * q
        rho_dcc[t] = q / np.sqrt(max(q**2, 0.001))

    rho_dcc = np.clip(rho_dcc, -0.99, 0.99)
    rho_dcc[0] = rho_dcc[1]

    return rho_dcc


# ============================================================================
# 5. 评估指标
# ============================================================================

def evaluate_estimation(rho_true, rho_est, name="Model"):
    """计算评估指标"""
    # 去除 NaN
    mask = ~(np.isnan(rho_est) | np.isnan(rho_true))
    rho_true_valid = rho_true[mask]
    rho_est_valid = rho_est[mask]

    mse = np.mean((rho_est_valid - rho_true_valid)**2)
    mae = np.mean(np.abs(rho_est_valid - rho_true_valid))
    rmse = np.sqrt(mse)

    # 相关系数
    corr = np.corrcoef(rho_true_valid, rho_est_valid)[0, 1]

    # 偏差
    bias = np.mean(rho_est_valid - rho_true_valid)

    print(f"\n{name} 评估结果:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  Corr: {corr:.4f}")
    print(f"  Bias: {bias:.4f}")

    return {
        'name': name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'corr': corr,
        'bias': bias
    }


# ============================================================================
# 6. 主模拟实验
# ============================================================================

def run_simulation_experiment(
    n=500,
    pattern='sinusoidal',
    marginal='normal',
    seed=42,
    verbose=True
):
    """
    运行单次模拟实验
    """
    print("=" * 70)
    print(f"模拟实验: n={n}, pattern={pattern}, marginal={marginal}")
    print("=" * 70)

    # 1. 模拟数据
    print("\n[1] 生成模拟数据...")
    rho_true = simulate_dynamic_correlation(n=n, pattern=pattern, seed=seed)
    x, y, rho_true = simulate_copula_data(
        n=n,
        rho_true=rho_true,
        marginal_x=marginal,
        marginal_y=marginal,
        seed=seed
    )

    print(f"  真实相关系数范围: [{rho_true.min():.3f}, {rho_true.max():.3f}]")
    print(f"  真实相关系数均值: {rho_true.mean():.3f}")

    # 2. 拟合各种模型
    print("\n[2] 拟合模型...")

    # GAS-Copula (MLE)
    start = time.time()
    gas_copula = SimpleGASCopula(ar=1, sc=1)
    gas_copula.fit(x, y, verbose=verbose)
    time_gas_copula = time.time() - start
    rho_gas_copula = gas_copula.get_dynamic_correlation()

    # GAS-Adam
    start = time.time()
    gas_adam = SimpleGASAdam(beta1=0.9, beta2=0.999, learning_rate=0.05, ar_coef=0.85)
    gas_adam.fit(x, y, verbose=verbose)
    time_gas_adam = time.time() - start
    rho_gas_adam = gas_adam.get_dynamic_correlation()

    # 滚动窗口
    start = time.time()
    rho_rolling = rolling_correlation(x, y, window=20)
    time_rolling = time.time() - start
    if verbose:
        print(f"滚动窗口相关系数计算完成")

    # DCC-GARCH
    start = time.time()
    rho_dcc = dcc_garch_simplified(x, y)
    time_dcc = time.time() - start
    if verbose:
        print(f"DCC-GARCH 计算完成")

    # 3. 评估
    print("\n[3] 模型评估...")
    results = []

    results.append(evaluate_estimation(rho_true, rho_gas_copula, "GAS-Copula"))
    results[-1]['time'] = time_gas_copula

    results.append(evaluate_estimation(rho_true, rho_gas_adam, "GAS-Adam"))
    results[-1]['time'] = time_gas_adam

    results.append(evaluate_estimation(rho_true, rho_rolling, "Rolling Window"))
    results[-1]['time'] = time_rolling

    results.append(evaluate_estimation(rho_true, rho_dcc, "DCC-GARCH"))
    results[-1]['time'] = time_dcc

    return {
        'rho_true': rho_true,
        'rho_gas_copula': rho_gas_copula,
        'rho_gas_adam': rho_gas_adam,
        'rho_rolling': rho_rolling,
        'rho_dcc': rho_dcc,
        'results': results,
        'x': x,
        'y': y
    }


def plot_comparison(exp_result, save_path=None):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rho_true = exp_result['rho_true']
    n = len(rho_true)
    t = np.arange(n)

    # 1. 相关系数路径对比
    ax = axes[0, 0]
    ax.plot(t, rho_true, 'k-', linewidth=2, label='真实值', alpha=0.8)
    ax.plot(t, exp_result['rho_gas_copula'], 'b-', linewidth=1.5, label='GAS-Copula', alpha=0.8)
    ax.plot(t, exp_result['rho_gas_adam'], 'r--', linewidth=1.5, label='GAS-Adam', alpha=0.8)
    ax.set_xlabel('时间')
    ax.set_ylabel('相关系数')
    ax.set_title('动态相关系数估计对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

    # 2. 所有方法对比
    ax = axes[0, 1]
    ax.plot(t, rho_true, 'k-', linewidth=2, label='真实值', alpha=0.7)
    ax.plot(t, exp_result['rho_rolling'], 'g-', linewidth=1, label='Rolling Window', alpha=0.7)
    ax.plot(t, exp_result['rho_dcc'], 'm-', linewidth=1, label='DCC-GARCH', alpha=0.7)
    ax.plot(t, exp_result['rho_gas_adam'], 'r-', linewidth=1.5, label='GAS-Adam', alpha=0.8)
    ax.set_xlabel('时间')
    ax.set_ylabel('相关系数')
    ax.set_title('与基准方法对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 误差分布
    ax = axes[1, 0]
    errors_gas_copula = exp_result['rho_gas_copula'] - rho_true
    errors_gas_adam = exp_result['rho_gas_adam'] - rho_true
    errors_rolling = exp_result['rho_rolling'] - rho_true

    ax.hist(errors_gas_copula, bins=30, alpha=0.5, label='GAS-Copula', density=True)
    ax.hist(errors_gas_adam, bins=30, alpha=0.5, label='GAS-Adam', density=True)
    ax.hist(errors_rolling[~np.isnan(errors_rolling)], bins=30, alpha=0.5, label='Rolling', density=True)
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel('估计误差')
    ax.set_ylabel('密度')
    ax.set_title('误差分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 性能对比条形图
    ax = axes[1, 1]
    names = [r['name'] for r in exp_result['results']]
    mses = [r['mse'] for r in exp_result['results']]
    colors = ['blue', 'red', 'green', 'purple']

    bars = ax.bar(names, mses, color=colors, alpha=0.7)
    ax.set_ylabel('MSE')
    ax.set_title('模型性能对比 (MSE)')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{mse:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至: {save_path}")

    # plt.show()  # 注释掉以在无 GUI 环境运行


def run_multiple_experiments(n_simulations=10, n=500):
    """运行多次实验并汇总结果"""
    print("=" * 70)
    print(f"运行 {n_simulations} 次蒙特卡洛实验")
    print("=" * 70)

    all_results = {
        'GAS-Copula': [],
        'GAS-Adam': [],
        'Rolling Window': [],
        'DCC-GARCH': []
    }

    patterns = ['sinusoidal', 'regime', 'gradual', 'volatile', 'mixed']

    for i in range(n_simulations):
        pattern = patterns[i % len(patterns)]
        seed = 42 + i

        print(f"\n--- 实验 {i+1}/{n_simulations} (pattern={pattern}) ---")

        result = run_simulation_experiment(
            n=n,
            pattern=pattern,
            seed=seed,
            verbose=False
        )

        for r in result['results']:
            all_results[r['name']].append(r)

    # 汇总统计
    print("\n" + "=" * 70)
    print("蒙特卡洛实验汇总结果")
    print("=" * 70)

    summary = {}
    for name, results in all_results.items():
        mses = [r['mse'] for r in results]
        maes = [r['mae'] for r in results]
        corrs = [r['corr'] for r in results]

        summary[name] = {
            'mse_mean': np.mean(mses),
            'mse_std': np.std(mses),
            'mae_mean': np.mean(maes),
            'corr_mean': np.mean(corrs)
        }

        print(f"\n{name}:")
        print(f"  MSE: {np.mean(mses):.6f} ± {np.std(mses):.6f}")
        print(f"  MAE: {np.mean(maes):.6f} ± {np.std(maes):.6f}")
        print(f"  Corr: {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")

    return summary


# ============================================================================
# 7. 主程序
# ============================================================================

if __name__ == '__main__':
    # 单次实验演示
    print("\n" + "=" * 70)
    print("GAS-Copula vs 其他模型 - 单次实验演示")
    print("=" * 70)

    # 运行实验
    exp_result = run_simulation_experiment(
        n=500,
        pattern='sinusoidal',
        marginal='normal',
        seed=42,
        verbose=True
    )

    # 绘制对比图
    plot_comparison(exp_result, save_path='/mnt/d/GAS模型重构/gas_copula_comparison.png')

    # 多次蒙特卡洛实验（可选，取消注释以运行）
    # print("\n" + "=" * 70)
    # print("运行蒙特卡洛实验...")
    # print("=" * 70)
    # mc_summary = run_multiple_experiments(n_simulations=5, n=300)
    #
    # # 绘制汇总对比图
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #
    # names = list(mc_summary.keys())
    # mse_means = [mc_summary[n]['mse_mean'] for n in names]
    # mse_stds = [mc_summary[n]['mse_std'] for n in names]
    #
    # # MSE 对比
    # ax = axes[0]
    # bars = ax.bar(names, mse_means, yerr=mse_stds, capsize=5,
    #                color=['blue', 'red', 'green', 'purple'], alpha=0.7)
    # ax.set_ylabel('MSE')
    # ax.set_title('蒙特卡洛实验: MSE 对比')
    # ax.grid(True, alpha=0.3, axis='y')
    #
    # # 相关系数对比
    # ax = axes[1]
    # corr_means = [mc_summary[n]['corr_mean'] for n in names]
    # bars = ax.bar(names, corr_means, color=['blue', 'red', 'green', 'purple'], alpha=0.7)
    # ax.set_ylabel('与真实值的相关系数')
    # ax.set_title('蒙特卡洛实验: 相关性对比')
    # ax.grid(True, alpha=0.3, axis='y')
    # ax.set_ylim(0, 1)
    #
    # plt.tight_layout()
    # plt.savefig('/mnt/d/GAS模型重构/monte_carlo_comparison.png', dpi=150, bbox_inches='tight')
    # plt.show()

    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
