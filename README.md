# Adam-GAS-Copula

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Adaptive Score-Driven Model for Dynamic Dependence Modeling with Adam Optimization**

使用 Adam 优化的 GAS (Generalized Autoregressive Score) 驱动的动态 Copula 相关系数建模。

---

## 概述

本项目实现了基于 GAS 模型的时变 Copula 相关系数建模方法，并引入 **Adam 优化器** 替代传统 MLE 估计，实现对二元变量之间动态依赖结构的灵活建模与对比分析。

### 核心特性

- **GAS-Copula 模型**: 基于得分驱动的动态相关系数建模
- **GAS-Adam 变体**: 使用 Adam 自适应优化器更新时变参数
- **多种 Copula 族**: Gaussian、Clayton、Gumbel Copula 支持
- **基准对比方法**: Rolling Window、DCC-GARCH
- **完整的模拟与评估框架**: 支持多种动态模式（正弦波、区制转换、渐变、高波动、混合）

---

## 项目结构

```
.
├── gas_copula_simulation.py    # 核心模拟与对比实验脚本
├── GAS_COPULA_DESIGN.md        # GAS-Copula 模型理论推导与设计文档
├── PIPELINE.md                 # PyFlux 重构流水线记录
├── pyflux/                     # 时间序列建模库 (重构版 v0.5.0)
├── Copulas-main/               # Copulas 概率建模库
├── adaptive_improved.png       # 自适应改进效果图
├── adaptive_vs_fixed.png       # 自适应 vs 固定参数对比图
├── gas_copula_comparison.png   # GAS-Copula 模型对比图
└── momentum_vs_gas.png         # Momentum vs GAS 对比图
```

---

## 快速开始

### 环境要求

- Python >= 3.10
- NumPy >= 1.24
- Pandas >= 2.0
- SciPy >= 1.15
- Matplotlib

### 安装

```bash
# 安装 pyflux
cd pyflux
pip install -e . --no-build-isolation

# 安装 copulas
cd Copulas-main/Copulas-main
pip install -e .
```

### 运行示例

```python
from gas_copula_simulation import run_simulation_experiment, plot_comparison

# 运行单次模拟实验
result = run_simulation_experiment(
    n=500,
    pattern='sinusoidal',  # 相关系数模式
    marginal='normal',     # 边缘分布
    seed=42
)

# 绘制对比图
plot_comparison(result, save_path='comparison.png')
```

---

## 方法对比

| 方法 | 类型 | 特点 |
|------|------|------|
| **GAS-Copula** | 得分驱动 | MLE 估计，Fisher 信息缩放 |
| **GAS-Adam** | 自适应优化 | Adam 更新规则，在线学习 |
| **Rolling Window** | 非参数基准 | 滚动窗口相关系数 |
| **DCC-GARCH** | 波动率模型 | 动态条件相关系数 |

---

## 理论背景

### GAS 模型

GAS 模型的核心更新方程：

$$f_{t+1} = \omega + A \cdot f_t + B \cdot S_t \nabla_t$$

其中 $\nabla_t$ 为得分函数，$S_t$ 为 Fisher 信息矩阵的逆作为缩放矩阵。

### Copula 密度

高斯 Copula 密度函数：

$$c(u, v; \rho) = \frac{1}{\sqrt{1-\rho^2}} \exp\left(-\frac{\rho^2(z_1^2 + z_2^2) - 2\rho z_1 z_2}{2(1-\rho^2)}\right)$$

### 相关系数约束

使用 tanh 变换保证相关系数在有效范围内：

$$\rho_t = \tanh(f_t), \quad f_t \in (-\infty, +\infty)$$

---

## 参考文献

1. Creal, D., Koopman, S. J., & Lucas, A. (2013). Generalized autoregressive score models with applications. *Journal of Applied Econometrics*, 28(5), 777-795.
2. Patton, A. J. (2006). Modelling asymmetric exchange rate dependence. *International Economic Review*, 47(2), 527-556.
3. Oh, D. H., & Patton, A. J. (2018). Time-varying systemic risk: Evidence from a dynamic copula model of CDS spreads. *Journal of Business & Economic Statistics*, 36(2), 181-195.
4. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.

---

## 许可证

MIT License
