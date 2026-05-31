# GAS-Copula 模型：动态相关系数建模

## 概述

本文档描述如何使用 GAS (Generalized Autoregressive Score) 模型来驱动二元 Copula 相关系数的动态更新，实现对两个变量 X 和 Y 之间依赖结构的时变建模。

---

## 1. 理论基础

### 1.1 Copula 简介

Copula 是一种将多变量联合分布分解为边缘分布和依赖结构的方法。根据 Sklar 定理：

$$F_{X,Y}(x,y) = C(F_X(x), F_Y(y); \theta)$$

其中：
- $F_{X,Y}$ 是联合分布函数
- $F_X, F_Y$ 是边缘分布函数
- $C$ 是 Copula 函数
- $\theta$ 是 Copula 参数（如相关系数 $\rho$）

### 1.2 二元高斯 Copula

对于二元高斯 Copula：

$$C(u, v; \rho) = \Phi_2(\Phi^{-1}(u), \Phi^{-1}(v); \rho)$$

其中：
- $u = F_X(x)$, $v = F_Y(y)$ 是通过概率积分变换得到的均匀分布变量
- $\Phi$ 是标准正态分布的 CDF
- $\Phi_2$ 是二元正态分布的 CDF
- $\rho$ 是相关系数参数

**Copula 密度函数**：

$$c(u, v; \rho) = \frac{1}{\sqrt{1-\rho^2}} \exp\left(-\frac{\rho^2(z_1^2 + z_2^2) - 2\rho z_1 z_2}{2(1-\rho^2)}\right)$$

其中 $z_1 = \Phi^{-1}(u)$, $z_2 = \Phi^{-1}(v)$。

### 1.3 GAS 模型原理

GAS 模型的核心更新方程：

$$f_{t+1} = \omega + \sum_{i=1}^{p} A_i f_{t-i+1} + \sum_{j=1}^{q} B_j S_t \nabla_t$$

其中：
- $f_t$ 是时变参数（这里是相关系数 $\rho_t$）
- $\nabla_t = \frac{\partial \log c(u_t, v_t; \rho_t)}{\partial f_t}$ 是得分函数
- $S_t$ 是缩放矩阵（通常选择为 Fisher 信息矩阵的逆）

---

## 2. GAS-Gaussian-Copula 实现

### 2.1 得分函数推导

对于二元高斯 Copula，对数似然函数：

$$\log c(u, v; \rho) = -\frac{1}{2}\log(1-\rho^2) - \frac{\rho^2(z_1^2 + z_2^2) - 2\rho z_1 z_2}{2(1-\rho^2)}$$

对 $\rho$ 求偏导（得分函数）：

$$\frac{\partial \log c}{\partial \rho} = \frac{\rho}{1-\rho^2} - \frac{\rho(z_1^2 + z_2^2) - (z_1 z_2)}{1-\rho^2} - \frac{\rho[\rho^2(z_1^2+z_2^2) - 2\rho z_1 z_2]}{(1-\rho^2)^2}$$

简化后：

$$\nabla_t = \frac{(z_1 z_2 - \rho)(1-\rho^2) + \rho(z_1^2 + z_2^2 - 2z_1 z_2 \rho)}{(1-\rho^2)^2}$$

### 2.2 缩放因子

使用 Fisher 信息矩阵的逆作为缩放因子：

$$\mathcal{I}(\rho) = E\left[\nabla^2\right] = \frac{1+\rho^2}{(1-\rho^2)^2}$$

$$S_t = \mathcal{I}(\rho)^{-1} = \frac{(1-\rho^2)^2}{1+\rho^2}$$

### 2.3 相关系数约束

相关系数必须在 $[-1, 1]$ 范围内。使用变换：

$$\rho_t = \tanh(f_t) = \frac{e^{f_t} - e^{-f_t}}{e^{f_t} + e^{-f_t}}$$

这样 $f_t \in (-\infty, +\infty)$，而 $\rho_t \in (-1, 1)$。

---

## 3. Python 实现代码

### 3.1 GAS-Copula 核心类

```python
import numpy as np
from scipy import stats
from scipy.special import ndtri

class GASCopula:
    """
    GAS驱动的二元高斯Copula模型
    
    用于动态建模两个变量之间的相关系数
    """
    
    def __init__(self, ar=1, sc=1):
        """
        Parameters
        ----------
        ar : int
            自回归阶数
        sc : int
            得分项阶数
        """
        self.ar = ar
        self.sc = sc
        
        # 模型参数: [omega, ar_coef..., sc_coef...]
        self.n_params = 1 + ar + sc
        self.params = None
        
    def _transform_to_uniform(self, x, y, x_dist, y_dist):
        """
        将数据变换到均匀分布空间
        
        Parameters
        ----------
        x, y : array-like
            原始数据
        x_dist, y_dist : scipy.stats distribution
            边缘分布对象
        """
        u = x_dist.cdf(x)
        v = y_dist.cdf(y)
        # 避免边界值
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        return u, v
    
    def _transform_to_normal(self, u, v):
        """
        将均匀变量变换到标准正态空间
        """
        z1 = ndtri(u)  # Phi^{-1}(u)
        z2 = ndtri(v)
        return z1, z2
    
    def _copula_log_likelihood(self, z1, z2, rho):
        """
        计算二元高斯Copula的对数似然
        
        Parameters
        ----------
        z1, z2 : array
            标准正态变换后的变量
        rho : float
            相关系数
        """
        if abs(rho) >= 1.0:
            return -np.inf
        
        term1 = -0.5 * np.log(1 - rho**2)
        term2 = -(rho**2 * (z1**2 + z2**2) - 2*rho*z1*z2) / (2 * (1 - rho**2))
        return term1 + term2
    
    def _score_function(self, z1, z2, rho):
        """
        计算得分函数（对数似然对rho的偏导）
        
        Parameters
        ----------
        z1, z2 : float
            标准正态变换后的变量
        rho : float
            当前相关系数
        """
        if abs(rho) >= 0.9999:
            rho = np.sign(rho) * 0.9999
        
        # 得分函数
        denom = (1 - rho**2)**2
        
        # 简化形式的得分
        numerator = (z1 * z2 - rho) * (1 - rho**2) + rho * (z1**2 + z2**2 - 2*z1*z2*rho)
        score = numerator / denom
        
        return score
    
    def _scaling_factor(self, rho):
        """
        计算缩放因子（Fisher信息矩阵的逆）
        """
        if abs(rho) >= 0.9999:
            rho = np.sign(rho) * 0.9999
        
        # Fisher信息的逆
        I_inv = (1 - rho**2)**2 / (1 + rho**2)
        return I_inv
    
    def _rho_to_f(self, rho):
        """
        将相关系数变换到无约束空间
        rho = tanh(f)
        """
        if abs(rho) >= 0.9999:
            rho = np.sign(rho) * 0.9999
        return np.arctanh(rho)
    
    def _f_to_rho(self, f):
        """
        将无约束变量变换回相关系数
        """
        return np.tanh(f)
    
    def fit(self, x, y, x_dist=None, y_dist=None, method='MLE', **kwargs):
        """
        拟合GAS-Copula模型
        
        Parameters
        ----------
        x, y : array-like
            观测数据
        x_dist, y_dist : scipy.stats distribution, optional
            边缘分布。如果为None，使用经验分布
        method : str
            估计方法：'MLE' 或 'PML'
        """
        from scipy.optimize import minimize
        from scipy.stats import norm
        
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        n = len(x)
        
        # 如果没有提供边缘分布，使用经验分布或正态分布
        if x_dist is None:
            x_dist = norm(loc=np.mean(x), scale=np.std(x))
        if y_dist is None:
            y_dist = norm(loc=np.mean(y), scale=np.std(y))
        
        # 变换到Copula空间
        u, v = self._transform_to_uniform(x, y, x_dist, y_dist)
        z1, z2 = self._transform_to_normal(u, v)
        
        self.z1 = z1
        self.z2 = z2
        self.u = u
        self.v = v
        
        # 目标函数
        def neg_log_likelihood(params):
            return self._compute_neg_ll(params, z1, z2)
        
        # 初始参数
        init_params = np.zeros(self.n_params)
        init_params[0] = 0.0  # omega
        init_params[1:1+self.ar] = 0.5 / self.ar  # AR系数
        init_params[1+self.ar:] = 0.3 / self.sc   # SC系数
        
        # 优化
        result = minimize(
            neg_log_likelihood,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        self.params = result.x
        self._compute_rho_path()
        
        return result
    
    def _compute_neg_ll(self, params, z1, z2):
        """
        计算负对数似然
        """
        n = len(z1)
        
        omega = params[0]
        ar_coefs = params[1:1+self.ar]
        sc_coefs = params[1+self.ar:1+self.ar+self.sc]
        
        max_lag = max(self.ar, self.sc)
        
        # 初始化
        f = np.zeros(n)  # 无约束的相关系数
        rho = np.zeros(n)  # 相关系数
        scores = np.zeros(n)
        
        # 初始值（无条件均值）
        f[:max_lag] = omega / (1 - np.sum(ar_coefs) + 1e-10)
        rho[:max_lag] = self._f_to_rho(f[:max_lag])
        
        total_ll = 0.0
        
        # GAS递归
        for t in range(max_lag, n):
            # 计算得分
            scores[t-1] = self._score_function(z1[t-1], z2[t-1], rho[t-1])
            scaled_score = self._scaling_factor(rho[t-1]) * scores[t-1]
            
            # GAS更新
            ar_term = np.sum(ar_coefs * f[t-self.ar:t][::-1]) if self.ar > 0 else 0
            sc_term = np.sum(sc_coefs * scores[t-self.sc:t][::-1]) if self.sc > 0 else 0
            
            f[t] = omega + ar_term + sc_term
            rho[t] = self._f_to_rho(f[t])
            
            # 累积对数似然
            ll = self._copula_log_likelihood(z1[t], z2[t], rho[t])
            if np.isfinite(ll):
                total_ll += ll
        
        return -total_ll
    
    def _compute_rho_path(self):
        """
        计算完整的相关系数路径
        """
        if self.params is None:
            return
        
        n = len(self.z1)
        omega = self.params[0]
        ar_coefs = self.params[1:1+self.ar]
        sc_coefs = self.params[1+self.ar:1+self.ar+self.sc]
        max_lag = max(self.ar, self.sc)
        
        self.f_path = np.zeros(n)
        self.rho_path = np.zeros(n)
        self.scores = np.zeros(n)
        
        # 初始值
        self.f_path[:max_lag] = omega / (1 - np.sum(ar_coefs) + 1e-10)
        self.rho_path[:max_lag] = self._f_to_rho(self.f_path[:max_lag])
        
        # GAS递归
        for t in range(max_lag, n):
            self.scores[t-1] = self._score_function(
                self.z1[t-1], self.z2[t-1], self.rho_path[t-1]
            )
            
            ar_term = np.sum(ar_coefs * self.f_path[t-self.ar:t][::-1]) if self.ar > 0 else 0
            sc_term = np.sum(sc_coefs * self.scores[t-self.sc:t][::-1]) if self.sc > 0 else 0
            
            self.f_path[t] = omega + ar_term + sc_term
            self.rho_path[t] = self._f_to_rho(self.f_path[t])
    
    def predict(self, h=1):
        """
        预测未来h期的相关系数
        
        Parameters
        ----------
        h : int
            预测期数
            
        Returns
        -------
        rho_forecast : array
            预测的相关系数
        """
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        omega = self.params[0]
        ar_coefs = self.params[1:1+self.ar]
        sc_coefs = self.params[1+self.ar:1+self.ar+self.sc]
        
        n = len(self.f_path)
        f_forecast = np.zeros(h)
        rho_forecast = np.zeros(h)
        
        # 使用最后的数据进行预测
        f_history = list(self.f_path[-max(self.ar, self.sc):])
        score_history = list(self.scores[-max(self.ar, self.sc):])
        
        for i in range(h):
            ar_term = np.sum(ar_coefs * np.array(f_history[-self.ar:])[::-1]) if self.ar > 0 else 0
            sc_term = np.sum(sc_coefs * np.array(score_history[-self.sc:])[::-1]) if self.sc > 0 else 0
            
            f_forecast[i] = omega + ar_term + sc_term
            rho_forecast[i] = self._f_to_rho(f_forecast[i])
            
            # 更新历史（预测时得分为0）
            f_history.append(f_forecast[i])
            score_history.append(0.0)
        
        return rho_forecast
    
    def get_dynamic_correlation(self):
        """
        获取估计的动态相关系数路径
        
        Returns
        -------
        rho_path : array
            时变相关系数
        """
        if self.rho_path is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.rho_path
```

### 3.2 使用示例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# =====================================================
# 示例 1: 模拟数据并拟合GAS-Copula模型
# =====================================================

np.random.seed(42)
n = 500

# 生成时变相关系数（真实值）
true_rho = np.zeros(n)
for t in range(n):
    true_rho[t] = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)  # 周期性变化

# 从二元正态分布生成数据
x = np.zeros(n)
y = np.zeros(n)

for t in range(n):
    rho_t = true_rho[t]
    # Cholesky分解
    L = np.array([[1, 0], [rho_t, np.sqrt(1 - rho_t**2)]])
    z = np.random.randn(2)
    xy = L @ z
    x[t] = xy[0]
    y[t] = xy[1]

# 拟合GAS-Copula模型
model = GASCopula(ar=1, sc=1)
model.fit(x, y)

# 获取估计的动态相关系数
estimated_rho = model.get_dynamic_correlation()

# 绘图比较
plt.figure(figsize=(12, 6))
plt.plot(true_rho, label='True ρ', alpha=0.7)
plt.plot(estimated_rho, label='Estimated ρ (GAS-Copula)', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.title('GAS-Copula: Dynamic Correlation Estimation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =====================================================
# 示例 2: 金融数据应用
# =====================================================

# 假设有股票收益率数据
# returns_stock1 = ...
# returns_stock2 = ...

# 拟合模型
# model = GASCopula(ar=1, sc=1)
# model.fit(returns_stock1, returns_stock2)

# 预测未来相关系数
# forecast_rho = model.predict(h=10)
# print(f"Forecasted correlation: {forecast_rho}")
```

---

## 4. 扩展：其他 Copula 族

### 4.1 Clayton Copula (下尾依赖)

Clayton Copula 更适合捕捉下尾依赖（熊市中的相关性增强）。

```python
class GASClaytonCopula:
    """
    GAS驱动的二元Clayton Copula模型
    
    Clayton Copula: C(u,v;θ) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}
    θ > 0，较大的θ表示更强的下尾依赖
    """
    
    def _copula_log_likelihood(self, u, v, theta):
        if theta <= 0:
            return -np.inf
        
        term1 = np.log(1 + theta)
        term2 = (-1 - theta) * (np.log(u) + np.log(v))
        term3 = -(1 + 2*theta) / theta * np.log(u**(-theta) + v**(-theta) - 1)
        
        return term1 + term2 + term3
    
    def _score_function(self, u, v, theta):
        # Clayton Copula的得分函数
        # ... 推导省略 ...
        pass
```

### 4.2 Gumbel Copula (上尾依赖)

Gumbel Copula 更适合捕捉上尾依赖（牛市中的相关性增强）。

```python
class GASGumbelCopula:
    """
    GAS驱动的二元Gumbel Copula模型
    
    Gumbel Copula: C(u,v;θ) = exp(-((-ln u)^θ + (-ln v)^θ)^{1/θ})
    θ >= 1，较大的θ表示更强的上尾依赖
    """
    pass
```

---

## 5. 模型诊断与验证

### 5.1 残差分析

```python
def diagnose_model(model):
    """
    对GAS-Copula模型进行诊断
    """
    # 获取概率积分变换后的残差
    u_residual = model.u
    v_residual = model.v
    
    # 应该服从均匀分布
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Q-Q图
    stats.probplot(u_residual, dist=stats.uniform, plot=axes[0])
    axes[0].set_title('Q-Q Plot: U residuals')
    
    stats.probplot(v_residual, dist=stats.uniform, plot=axes[1])
    axes[1].set_title('Q-Q Plot: V residuals')
    
    plt.tight_layout()
    plt.show()
    
    # Kolmogorov-Smirnov检验
    ks_u = stats.kstest(u_residual, 'uniform')
    ks_v = stats.kstest(v_residual, 'uniform')
    
    print(f"KS Test for U: statistic={ks_u.statistic:.4f}, p-value={ks_u.pvalue:.4f}")
    print(f"KS Test for V: statistic={ks_v.statistic:.4f}, p-value={ks_v.pvalue:.4f}")
```

### 5.2 滚动窗口验证

```python
def rolling_validation(x, y, window_size=100, forecast_horizon=1):
    """
    滚动窗口验证
    """
    n = len(x)
    forecasts = []
    actuals = []
    
    for i in range(window_size, n - forecast_horizon):
        # 训练窗口
        x_train = x[i-window_size:i]
        y_train = y[i-window_size:i]
        
        # 拟合模型
        model = GASCopula(ar=1, sc=1)
        model.fit(x_train, y_train)
        
        # 预测
        rho_forecast = model.predict(h=forecast_horizon)
        forecasts.append(rho_forecast[0])
        
        # 实际相关系数（使用简单移动窗口估计）
        actual_rho = np.corrcoef(x[i:i+forecast_horizon], 
                                  y[i:i+forecast_horizon])[0, 1]
        actuals.append(actual_rho)
    
    # 计算预测误差
    forecasts = np.array(forecasts)
    actuals = np.array(actuals)
    rmse = np.sqrt(np.mean((forecasts - actuals)**2))
    
    print(f"Rolling Window RMSE: {rmse:.4f}")
    return forecasts, actuals
```

---

## 6. 与 PyFlux 集成

### 6.1 作为 PyFlux 新模块

```python
# 在 pyflux/gas/ 目录下添加 gas_copula.py

from ..tsm import TSM
from .. import families as fam

class GASCopula(TSM):
    """
    GAS驱动的二元Copula模型
    
    继承自TSM基类，支持MLE、PML、M-H、BBVI等推断方法
    """
    
    def __init__(self, x, y, ar=1, sc=1, copula_type='gaussian'):
        super().__init__('GASCopula')
        
        self.x = x
        self.y = y
        self.ar = ar
        self.sc = sc
        self.copula_type = copula_type
        
        self.z_no = 1 + ar + sc  # 参数数量
        self.supported_methods = ["MLE", "PML", "M-H", "BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False
        
        self._create_latent_variables()
    
    def _create_latent_variables(self):
        """创建潜变量"""
        # omega
        self.latent_variables.add_z('omega', fam.Normal(0, 1), fam.Normal(0, 3))
        
        # AR coefficients
        for i in range(self.ar):
            self.latent_variables.add_z(
                f'ar_{i+1}', 
                fam.Normal(0, 0.5, transform='logit'), 
                fam.Normal(0, 3)
            )
        
        # SC coefficients  
        for j in range(self.sc):
            self.latent_variables.add_z(
                f'sc_{j+1}', 
                fam.Normal(0, 0.5), 
                fam.Normal(0, 3)
            )
    
    def neg_loglik(self, beta):
        """计算负对数似然"""
        # ... 实现详见上文 ...
        pass
    
    def _model(self, beta):
        """模型递归"""
        # ... GAS递归实现 ...
        pass
```

---

## 7. 参考文献

1. **GAS模型**:
   - Creal, D., Koopman, S. J., & Lucas, A. (2013). Generalized autoregressive score models with applications. *Journal of Applied Econometrics*, 28(5), 777-795.

2. **动态Copula**:
   - Patton, A. J. (2006). Modelling asymmetric exchange rate dependence. *International Economic Review*, 47(2), 527-556.

3. **GAS-Copula**:
   - Oh, D. H., & Patton, A. J. (2018). Time-varying systemic risk: Evidence from a dynamic copula model of CDS spreads. *Journal of Business & Economic Statistics*, 36(2), 181-195.

---

*文档版本: 1.0 | 创建日期: 2026-04-22*
