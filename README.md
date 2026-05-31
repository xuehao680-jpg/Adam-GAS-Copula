# Adam-GAS-Copula

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**GAS-Driven Dynamic Copula Correlation Modeling with Adam Optimization**

---

## 目录

- [背景与动机](#背景与动机)
- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [理论背景](#理论背景)
- [模型详解](#模型详解)
- [环境要求](#环境要求)
- [安装与构建](#安装与构建)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [实验设计](#实验设计)
- [结果可视化](#结果可视化)
- [评估指标](#评估指标)
- [对比方法](#对比方法)
- [PyFlux 重构说明](#pyflux-重构说明)
- [Copulas 库说明](#copulas-库说明)
- [参考文献](#参考文献)
- [引用](#引用)
- [许可证](#许可证)

---

## 背景与动机

金融市场中，不同资产之间的相关性并非恒定不变。例如：
- 金融危机期间，股票之间的相关性往往会急剧上升（"相关性崩溃"）
- 牛市和熊市中，资产间的依赖结构可能存在不对称性（上尾依赖 vs 下尾依赖）
- 宏观政策变化、行业周期等因素会导致资产关联关系发生结构性变化

传统的静态相关系数（如 Pearson 相关系数）无法捕捉这些时变特征。本项目结合两大前沿方法：

1. **GAS (Generalized Autoregressive Score) 模型** — 一种灵活的时变参数驱动框架，利用得分函数（score function）驱动参数更新
2. **Copula 函数** — 将多变量联合分布分解为边缘分布和依赖结构的数学工具

并在此基础上引入 **Adam 自适应优化器**，为时变 Copula 相关系数建模提供了一种新的在线学习范式。

---

## 项目概述

本项目实现了一套完整的 **时变 Copula 动态相关系数建模与对比分析框架**，主要包含：

| 模块 | 说明 |
|------|------|
| **GAS-Copula** | 基于得分驱动 + MLE 估计的传统 GAS-Copula 模型 |
| **GAS-Adam** | 使用 Adam 优化器在线更新时变参数的自适应变体 |
| **基准方法** | Rolling Window、DCC-GARCH 等传统方法作为对照组 |
| **数据模拟** | 5 种动态相关系数模式（正弦波、区制转换、渐变、高波动、混合） |
| **评估框架** | MSE、RMSE、MAE、相关性、偏差等多维度评估 |
| **PyFlux** | 完整重构的时间序列建模库（v0.5.0，Python 3.10+ 兼容） |
| **Copulas** | 概率建模与合成数据生成库 |

---

## 核心特性

### 1. GAS-Copula 模型 (MLE)

- 利用 GAS 框架驱动二元 Copula 相关系数 $\rho_t$ 的时变更新
- 得分函数基于对数 Copula 密度的梯度推导
- Fisher 信息矩阵作为自适应缩放因子
- L-BFGS-B 数值优化器进行全参数估计
- 支持 AR(p) 和 SC(q) 阶数配置

### 2. GAS-Adam 变体 (在线学习)

- 不依赖批量 MLE 优化，而是逐时间步在线更新
- Adam 自适应矩估计替代传统得分缩放
- 可配置的学习率、$\beta_1$、$\beta_2$ 参数
- 无需数值优化器即可拟合，计算效率更高
- 适合数据流、实时更新等在线学习场景

### 3. 多种 Copula 族支持

| Copula 类型 | 尾部依赖特征 | 适用场景 |
|-------------|-------------|----------|
| **Gaussian** | 无尾部依赖 | 一般性对称依赖 |
| **Clayton** | 下尾依赖 | 熊市中相关性增强 |
| **Gumbel** | 上尾依赖 | 牛市中相关性增强 |

### 4. 完整的模拟与评估框架

- 5 种动态相关系数模式
- 多边缘分布支持（Normal, Student-t, Exponential, Uniform）
- 单次实验 + 蒙特卡洛多次实验
- 可视化对比图自动生成

---

## 项目结构

```
Adam-GAS-Copula/
├── README.md                      # 项目说明文档
├── GAS_COPULA_DESIGN.md            # GAS-Copula 模型完整理论推导
├── PIPELINE.md                     # PyFlux 重构流水线详细记录
│
├── gas_copula_simulation.py        # ★ 核心模拟与对比实验脚本
│   ├── simulate_dynamic_correlation()  # 5种模式数据生成
│   ├── simulate_copula_data()          # Copula 双变量数据模拟
│   ├── SimpleGASCopula                 # GAS-Copula (MLE) 实现
│   ├── SimpleGASAdam                   # GAS-Adam (在线学习) 实现
│   ├── rolling_correlation()           # 滚动窗口基准方法
│   ├── dcc_garch_simplified()          # DCC-GARCH 基准方法
│   ├── evaluate_estimation()           # 评估指标计算
│   ├── run_simulation_experiment()     # 单次实验主函数
│   ├── run_multiple_experiments()      # 蒙特卡洛实验
│   └── plot_comparison()               # 4面板对比图
│
├── pyflux/                         # 时间序列建模库 (重构版 v0.5.0)
│   ├── pyproject.toml              # 现代构建配置
│   ├── setup.py                    # Cython 扩展编译
│   ├── gas/                        # GAS 模型模块
│   │   ├── gas.py                  # GAS 正态模型
│   │   ├── gasx.py                 # GAS 外生变量模型
│   │   ├── gasllm.py               # GAS 局部水平模型
│   │   ├── gasllt.py               # GAS 局部线性趋势模型
│   │   ├── gasreg.py               # GAS 回归模型
│   │   ├── gasrank.py              # GAS 秩模型
│   │   └── gas_core_recursions.c   # Cython 核心递归
│   ├── arma/                       # ARMA/ARIMA 模型模块
│   ├── garch/                      # GARCH 波动率模型模块
│   ├── ssm/                        # 状态空间模型模块
│   ├── var/                        # 向量自回归模型模块
│   ├── gpnarx/                     # 高斯过程模块
│   ├── inference/                  # 推断方法 (MLE/M-H/BBVI)
│   └── families/                   # 概率分布族
│
├── Copulas-main/                   # Copulas 概率建模库
│   └── Copulas-main/
│       ├── copulas/                # 核心库
│       │   ├── bivariate/          # 二元 Copula (Clayton, Frank, Gumbel)
│       │   ├── multivariate/       # 多元 Copula (Gaussian, Vine)
│       │   ├── univariate/         # 单变量分布
│       │   └── optimize/           # 优化模块
│       ├── tests/                  # 测试套件
│       │   ├── unit/               # 单元测试
│       │   ├── end-to-end/         # 端到端测试
│       │   └── numerical/          # 数值精度测试
│       └── tutorials/              # Jupyter 教程
│
└── *.png                           # 实验结果可视化图
    ├── gas_copula_comparison.png   # GAS-Copula 模型对比 (4面板)
    ├── adaptive_improved.png       # 自适应改进效果
    ├── adaptive_vs_fixed.png       # 自适应 vs 固定参数
    └── momentum_vs_gas.png         # Momentum vs GAS 对比
```

---

## 理论背景

### Sklar 定理与 Copula

根据 Sklar 定理 (1959)，任何二元联合分布函数 $F_{X,Y}(x,y)$ 可以分解为：

$$F_{X,Y}(x,y) = C(F_X(x), F_Y(y); \theta)$$

其中：
- $F_X, F_Y$ 是边缘分布函数
- $C: [0,1]^2 \to [0,1]$ 是 Copula 函数
- $\theta$ 是 Copula 参数（如相关系数 $\rho$）

这一分解的革命性意义在于：**边缘分布的选择和依赖结构的选择可以完全独立**。

### 二元高斯 Copula

对于二元高斯 Copula：

$$C(u, v; \rho) = \Phi_2(\Phi^{-1}(u), \Phi^{-1}(v); \rho)$$

其中：
- $u = F_X(x), v = F_Y(y)$ 是概率积分变换后的均匀变量
- $\Phi$ 为标准正态 CDF
- $\Phi_2$ 为二元正态 CDF
- $\rho \in (-1, 1)$ 为相关系数

**Copula 密度函数**：

$$c(u, v; \rho) = \frac{1}{\sqrt{1-\rho^2}} \exp\left(-\frac{\rho^2(z_1^2 + z_2^2) - 2\rho z_1 z_2}{2(1-\rho^2)}\right)$$

其中 $z_1 = \Phi^{-1}(u)$, $z_2 = \Phi^{-1}(v)$。

### GAS 模型框架

GAS 模型由 Creal, Koopman & Lucas (2013) 提出，其时变参数更新机制为：

$$f_{t+1} = \omega + \sum_{i=1}^{p} A_i f_{t-i+1} + \sum_{j=1}^{q} B_j S_t \nabla_t$$

其中：

| 符号 | 含义 |
|------|------|
| $f_t$ | 无约束空间的时变参数 |
| $\nabla_t = \frac{\partial \log p(y_t | f_t)}{\partial f_t}$ | 得分函数 (Score) |
| $S_t$ | 缩放矩阵（通常为 Fisher 信息矩阵的逆） |
| $\omega$ | 截距项（长期均值） |
| $A_i$ | 自回归系数 |
| $B_j$ | 得分项系数 |

**核心直觉**：GAS 模型沿着对数似然梯度方向更新参数——如果当前参数值下观测数据的似然梯度很陡，说明参数需要大幅调整；如果梯度平缓，参数保持稳定。

### 得分函数推导

对二元高斯 Copula 的对数密度求导，得到得分函数：

$$\frac{\partial \log c}{\partial \rho} = \frac{(z_1 z_2 - \rho)(1-\rho^2) + \rho(z_1^2 + z_2^2 - 2z_1 z_2 \rho)}{(1-\rho^2)^2}$$

### Fisher 信息缩放

Fisher 信息矩阵衡量得分函数的方差：

$$\mathcal{I}(\rho) = E[\nabla^2] = \frac{1+\rho^2}{(1-\rho^2)^2}$$

缩放因子为其逆：$S_t = \mathcal{I}(\rho)^{-1} = \frac{(1-\rho^2)^2}{1+\rho^2}$

### 相关系数约束

使用 tanh 变换确保相关系数始终在有效区间：

$$\rho_t = \tanh(f_t) = \frac{e^{f_t} - e^{-f_t}}{e^{f_t} + e^{-f_t}}, \quad \rho_t \in (-1, 1)$$

---

## 模型详解

### GAS-Copula (MLE 估计)

```python
class SimpleGASCopula:
    """
    GAS 驱动的二元高斯 Copula 模型

    Parameters
    ----------
    ar : int, default=1
        自回归阶数 p
    sc : int, default=1
        得分项阶数 q

    估计方法
    --------
    - 通过 L-BFGS-B 优化器最小化负对数似然
    - 参数: omega, AR 系数, SC 系数
    - 递归计算时变 rho_t 路径

    核心步骤
    --------
    1. 边缘分布估计 -> 概率积分变换 -> u, v
    2. 逆正态变换 -> z1, z2
    3. GAS 递归: f_t -> rho_t = tanh(f_t)
    4. 得分计分: score_t = d(log c)/d(rho)
    5. 参数更新: f_{t+1} = omega + A*f_t + B*S_t*score_t
    """
```

### GAS-Adam (在线学习)

```python
class SimpleGASAdam:
    """
    使用 Adam 优化器在线更新时变 Copula 参数的模型

    Parameters
    ----------
    beta1 : float, default=0.9
        一阶矩衰减率
    beta2 : float, default=0.999
        二阶矩衰减率
    learning_rate : float, default=0.05
        全局学习率
    ar_coef : float, default=0.85
        自回归系数 A

    核心更新规则
    ------------
    m_t = beta1 * m_{t-1} + (1-beta1) * score_t
    v_t = beta2 * v_{t-1} + (1-beta2) * score_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    f_{t+1} = omega + A * f_t + lr * m_hat / (sqrt(v_hat) + eps)
    rho_{t+1} = tanh(f_{t+1})

    与 GAS-Copula 的关键区别
    -------------------------
    - 不使用批量 MLE，逐时间步在线更新
    - Adam 自适应学习率替代 Fisher 信息缩放
    - 不需要数值优化器
    - 支持非平稳环境的快速适应
    """
```

---

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+ | 需要现代 typing 和 match 语法 |
| NumPy | 1.24+ | `np.float64` 替代已弃用的 `np.float` |
| Pandas | 2.0+ | `.iloc[]` 替代已弃用的 `.ix[]` |
| SciPy | 1.15+ | L-BFGS-B 优化、统计分布 |
| Matplotlib | 3.5+ | 可视化 |
| Cython | 3.0+ | (可选) PyFlux C 扩展编译 |
| Seaborn | 0.12+ | (可选) KDE 图 |

---

## 安装与构建

### 1. 克隆仓库

```bash
git clone https://github.com/xuehao680-jpg/Adam-GAS-Copula.git
cd Adam-GAS-Copula
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
```

### 3. 安装核心依赖

```bash
pip install numpy scipy pandas matplotlib seaborn
```

### 4. 安装 PyFlux（可选，含 Cython 扩展）

```bash
cd pyflux
pip install -e . --no-build-isolation

# 验证 Cython 扩展
python -c "
from pyflux.arma.arma_recursions import arima_recursion
from pyflux.gas.gas_core_recursions import gas_recursion
from pyflux.ssm.kalman import univariate_KFS
print('All Cython extensions loaded successfully')
"
```

### 5. 安装 Copulas（可选）

```bash
cd Copulas-main/Copulas-main
pip install -e .
```

---

## 快速开始

### 最简示例

```python
from gas_copula_simulation import (
    run_simulation_experiment,
    plot_comparison
)

# 运行一次完整的模拟实验
result = run_simulation_experiment(
    n=500,                        # 样本量
    pattern='sinusoidal',          # 相关系数模式
    marginal='normal',             # 边缘分布类型
    seed=42                        # 随机种子
)

# 生成 4 面板对比图
plot_comparison(result, save_path='gas_copula_comparison.png')
```

**输出**：
- `result['rho_true']` — 真实相关系数路径
- `result['rho_gas_copula']` — GAS-Copula 估计值
- `result['rho_gas_adam']` — GAS-Adam 估计值
- `result['rho_rolling']` — 滚动窗口估计值
- `result['rho_dcc']` — DCC-GARCH 估计值
- `result['results']` — 各方法的 MSE/RMSE/MAE/Corr/Bias 评估指标

### 自定义数据

```python
import numpy as np
from gas_copula_simulation import SimpleGASCopula, SimpleGASAdam

# 使用你自己的数据
x = np.array([...])  # 变量 X 的观测值
y = np.array([...])  # 变量 Y 的观测值

# GAS-Copula 拟合
gas_model = SimpleGASCopula(ar=1, sc=1)
gas_model.fit(x, y)
rho_gas = gas_model.get_dynamic_correlation()

# GAS-Adam 拟合
adam_model = SimpleGASAdam(
    beta1=0.9,
    beta2=0.999,
    learning_rate=0.05,
    ar_coef=0.85
)
adam_model.fit(x, y)
rho_adam = adam_model.get_dynamic_correlation()
```

### 蒙特卡洛实验

```python
from gas_copula_simulation import run_multiple_experiments

# 运行 10 次蒙特卡洛实验
summary = run_multiple_experiments(n_simulations=10, n=500)
# 输出各种相关模式的汇总 MSE、MAE、相关性等统计量
```

---

## API 参考

### SimpleGASCopula

```python
class SimpleGASCopula(ar=1, sc=1):
    """
    GAS-Copula 模型 (MLE)

    Methods
    -------
    fit(x, y, verbose=True)
        使用 L-BFGS-B 拟合模型。
        自动进行秩变换 -> 均匀分布 -> 逆正态变换。

    get_dynamic_correlation()
        返回完整的时变相关系数路径。

    Attributes
    ----------
    params : np.ndarray
        估计的参数 [omega, ar_coefs..., sc_coefs...]
    rho_path : np.ndarray
        时变相关系数序列
    scores : np.ndarray
        各时间步的得分函数值
    ll : float
        模型对数似然值
    """
```

### SimpleGASAdam

```python
class SimpleGASAdam(beta1=0.9, beta2=0.999, learning_rate=0.05, ar_coef=0.85):
    """
    GAS-Adam 模型 (在线学习)

    Methods
    -------
    fit(x, y, verbose=True)
        逐时间步在线拟合，无需数值优化器。

    get_dynamic_correlation()
        返回完整的时变相关系数路径。

    Attributes
    ----------
    rho_path : np.ndarray
        时变相关系数序列
    total_nll : float
        总负对数似然损失
    omega : float
        截距参数
    """
```

### 数据生成函数

```python
def simulate_dynamic_correlation(n=500, pattern='sinusoidal', seed=None, **kwargs):
    """
    生成时变相关系数真实路径。

    pattern 选项:
        - 'sinusoidal' : 正弦波动 (支持 period 参数)
        - 'regime'      : 区制转换（低/高/零/中相关）
        - 'gradual'     : 线性渐变
        - 'volatile'    : 高波动随机游走
        - 'mixed'       : 趋势 + 周期 + 随机噪声
    """

def simulate_copula_data(n=500, rho_true=None, marginal_x='normal',
                         marginal_y='normal', seed=None):
    """
    基于 Copula 模拟双变量数据。

    marginal 选项: 'normal', 't', 'exponential', 'uniform'
    """

# 基准方法
def rolling_correlation(x, y, window=20):
    """滚动窗口相关系数（非参数基准）"""

def dcc_garch_simplified(x, y, init_alpha=0.05, init_beta=0.9):
    """简化版 DCC-GARCH 动态相关系数"""

# 评估函数
def evaluate_estimation(rho_true, rho_est, name="Model"):
    """计算 MSE, RMSE, MAE, Corr, Bias 等指标"""
```

---

## 实验设计

### 动态相关系数模式

| 模式 | 数学形式 | 特点 |
|------|---------|------|
| **sinusoidal** | $\rho_t = 0.3 + 0.4\sin(2\pi t / 100)$ | 周期性波动 |
| **regime** | 分4段：0.2 → 0.7 → 0.0 → 0.5 | 结构性跃变 |
| **gradual** | $\rho_t = 0.1 + 0.6t/n$ | 单调趋势 |
| **volatile** | 随机游走累积和 | 高噪声 |
| **mixed** | 趋势 + 正弦 + 随机游走 | 复合模式 |

### 数据生成流程

```
真实 rho_t 路径
    ↓
二元正态模拟 (Cholesky 分解)
    ↓
边缘分布变换 (Normal/t/Exp/Uniform)
    ↓
观测数据 (x_t, y_t)
```

### 模型对比流程

```
观测数据 (x, y)
    ↓         ↓         ↓         ↓
GAS-Copula  GAS-Adam  Rolling   DCC-GARCH
 (MLE)      (在线)    (20期)    (alpha, beta)
    ↓         ↓         ↓         ↓
        评估与可视化对比
```

---

## 结果可视化

`plot_comparison()` 函数生成 4 面板对比图：

1. **左上 - GAS 方法对比**：真实值 vs GAS-Copula vs GAS-Adam
2. **右上 - 基准方法对比**：真实值 vs Rolling Window vs DCC-GARCH vs GAS-Adam
3. **左下 - 误差分布**：各方法估计误差的直方图（附核密度）
4. **右下 - 性能对比**：各方法 MSE 条形图

---

## 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(\hat{\rho}_t - \rho_t)^2$ | 均方误差 |
| RMSE | $\sqrt{MSE}$ | 均方根误差 |
| MAE | $\frac{1}{n}\sum|\hat{\rho}_t - \rho_t|$ | 平均绝对误差 |
| Corr | $corr(\hat{\rho}, \rho)$ | 估计值与真实值的相关性 |
| Bias | $\frac{1}{n}\sum(\hat{\rho}_t - \rho_t)$ | 系统性偏差 |

---

## 对比方法

### 1. 滚动窗口相关系数 (Rolling Window)

- 固定窗口大小（默认 20 期）
- 计算窗口内 Pearson 相关系数
- 优点：简单、无模型假设
- 缺点：滞后严重、窗宽敏感、边界不平滑

### 2. DCC-GARCH (Dynamic Conditional Correlation)

- 使用 EWMA ($\lambda=0.94$) 估计波动率
- DCC(1,1) 参数化动态相关性
- 优点：经典方法、文献丰富
- 缺点：Gaussian 假设、参数敏感

### 3. GAS-Copula (MLE)

- 得分驱动 + Fisher 信息缩放
- 全局 MLE 参数估计
- 优点：统计最优、解析得分
- 缺点：需批量数据、数值优化耗时

### 4. GAS-Adam (在线)

- Adam 自适应矩估计
- 逐时间步在线更新
- 优点：无需批量优化、快速适应
- 缺点：超参数敏感、非统计最优

---

## PyFlux 重构说明

本仓库包含的 `pyflux/` 目录是原始 PyFlux 时间序列库的全量重构版本（v0.5.0），主要改动：

### 阶段 1：基础设施迁移
- `numpy.distutils` → 现代 `setuptools` + `pyproject.toml`
- 删除各子模块的独立 `setup.py`

### 阶段 2：Python 2 代码清理
- 移除 30 个文件中的 `xrange` 兼容代码
- 清理 `sys.version_info` 检查

### 阶段 3：NumPy API 更新
- `np.float` → `np.float64`（17 处替换）

### 阶段 4：Pandas API 更新
- `.ix[]` → `.iloc[]`
- `pd.Int64Index` → `pd.Index`

### 阶段 5：Seaborn/Super 更新
- `sns.distplot` → `sns.kdeplot` / `sns.histplot`（~20 处）
- `super(Class, self).__init__()` → `super().__init__()`（~15 处）

### 最终测试环境

```
Python:   3.10.20
NumPy:    2.2.6
Pandas:   2.3.3
SciPy:    1.15.3
Cython:   3.2.4
GAS 测试:  24/24 PASS
ARIMA 测试: 17/17 PASS
GARCH 测试: 全部通过
```

详细记录见 [PIPELINE.md](PIPELINE.md)。

---

## Copulas 库说明

本仓库包含的 `Copulas-main/` 目录是 SDV 生态下的概率建模库，提供了丰富的 Copula 模型实现：

- **二元 Copula**: Clayton、Frank、Gumbel、Independence
- **多元 Copula**: Gaussian、Vine Copula（C-Vine/D-Vine）
- **单变量分布**: Gaussian、Student-t、Beta、Gamma、Gaussian KDE、Truncated Gaussian
- **优化**: 基于 `scipy.optimize` 的参数估计
- **可视化**: 分布对比、PIT 直方图、CDF/PDF 图

该库的测试套件完整（单元测试 + 端到端测试 + 数值精度测试），可用于合成数据生成和概率建模场景。

---

## 参考文献

1. **Creal, D., Koopman, S. J., & Lucas, A.** (2013). Generalized autoregressive score models with applications. *Journal of Applied Econometrics*, 28(5), 777–795.

2. **Harvey, A. C.** (2013). *Dynamic Models for Volatility and Heavy Tails: With Applications to Financial and Economic Time Series*. Cambridge University Press.

3. **Patton, A. J.** (2006). Modelling asymmetric exchange rate dependence. *International Economic Review*, 47(2), 527–556.

4. **Oh, D. H., & Patton, A. J.** (2018). Time-varying systemic risk: Evidence from a dynamic copula model of CDS spreads. *Journal of Business & Economic Statistics*, 36(2), 181–195.

5. **Kingma, D. P., & Ba, J.** (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*.

6. **Sklar, A.** (1959). Fonctions de répartition à n dimensions et leurs marges. *Publications de l'Institut de Statistique de l'Université de Paris*, 8, 229–231.

7. **Engle, R. F.** (2002). Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. *Journal of Business & Economic Statistics*, 20(3), 339–350.

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{adam-gas-copula,
  author       = {Xue Hao},
  title        = {Adam-GAS-Copula: Adaptive Score-Driven Model for Dynamic Dependence Modeling},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/xuehao680-jpg/Adam-GAS-Copula}}
}
```

---

## 许可证

本项目基于 MIT License 开源。详见 [LICENSE](LICENSE) 文件。
