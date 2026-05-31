# PyFlux 重构 PIPELINE

## 版本信息
- 开始日期: 2026-04-22
- 完成日期: 2026-04-22
- 目标版本: 0.5.0
- **Python 支持**: >=3.10
- **NumPy 支持**: >=1.24
- **Pandas 支持**: >=2.0
- **重构范围**: 全部模块 (gas, arma, garch, ssm, var, gpnarx, inference, families)

---

## 阶段进度总览

| 阶段 | 名称 | 状态 | 测试结果 |
|------|------|------|----------|
| 1 | 基础设施更新 | ✅ 完成 | PASS |
| 2 | Python 2 代码清理 | ✅ 完成 | PASS |
| 3 | NumPy 类型修复 | ✅ 完成 | PASS |
| 4 | Pandas API 修复 | ✅ 完成 | PASS |
| 5 | Seaborn/Super 更新 | ✅ 完成 | PASS |

---

## 阶段 1: 基础设施更新

### 目标
迁移构建系统从 `numpy.distutils` 到现代 `setuptools` + `pyproject.toml`

### 检查清单
- [x] 创建 `pyproject.toml`
- [x] 重写主 `setup.py`
- [x] 更新 `requirements.txt`
- [x] 删除/更新子模块 `setup.py`
- [x] 验证 Cython 扩展编译

### 修改文件列表
| 文件 | 操作 |
|------|------|
| `pyproject.toml` | 新建 |
| `setup.py` | 重写 |
| `requirements.txt` | 更新版本 |
| `pyflux/setup.py` | 删除/简化 |
| `pyflux/arma/setup.py` | 删除 |
| `pyflux/families/setup.py` | 删除 |
| `pyflux/garch/setup.py` | 删除 |
| `pyflux/gas/setup.py` | 删除 |
| `pyflux/gpnarx/setup.py` | 删除 |
| `pyflux/inference/setup.py` | 删除 |
| `pyflux/ssm/setup.py` | 删除 |
| `pyflux/var/setup.py` | 删除 |

### MVP 测试命令
```bash
# 创建测试环境
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# test_env\Scripts\activate   # Windows

# 安装依赖
pip install --upgrade pip setuptools wheel cython numpy

# 构建安装
pip install -e . --no-build-isolation -v

# 验证 Cython 扩展
python -c "
from pyflux.arma.arma_recursions import arima_recursion
from pyflux.gas.gas_core_recursions import gas_recursion
from pyflux.ssm.kalman import univariate_KFS
print('All Cython extensions loaded successfully')
"

# 验证基础导入
python -c "import pyflux; print('pyflux version:', pyflux.__version__)"
```

### 测试结果记录
```
日期: 2026-04-22
Python 版本: 3.10.20
NumPy 版本: 2.2.6
Cython 扩展: 全部编译成功 (12个模块)
基础导入: 成功
错误信息: 无
```

---

## 阶段 2: Python 2 兼容代码清理

### 目标
移除所有 Python 2 兼容代码

### 检查清单
- [x] 移除 xrange 兼容代码 (30个文件)
- [x] 清理 `sys.version_info` 检查
- [x] 验证无残留 Python 2 代码

### 修改文件列表
```
pyflux/tsm.py
pyflux/inference/bbvi.py
pyflux/inference/norm_post_sim.py
pyflux/inference/metropolis_hastings.py
pyflux/gas/gas.py
pyflux/gas/gasx.py
pyflux/gas/gasllm.py
pyflux/gas/gasllt.py
pyflux/gas/gasreg.py
pyflux/gas/gasrank.py
pyflux/arma/arma.py
pyflux/arma/arimax.py
pyflux/arma/nnar.py
pyflux/arma/nnarx.py
pyflux/garch/garch.py
pyflux/garch/egarch.py
pyflux/garch/egarchm.py
pyflux/garch/egarchmreg.py
pyflux/garch/segarch.py
pyflux/garch/segarchm.py
pyflux/garch/lmegarch.py
pyflux/ssm/llm.py
pyflux/ssm/llt.py
pyflux/ssm/nllm.py
pyflux/ssm/nllt.py
pyflux/ssm/dar.py
pyflux/ssm/dynlin.py
pyflux/ssm/ndynlin.py
pyflux/var/var.py
pyflux/gpnarx/gpnarx.py
```

### 代码修改示例
```python
# 删除以下代码块 (通常在文件开头):
import sys
if sys.version_info < (3,):
    range = xrange
```

### MVP 测试命令
```bash
# 检查无 xrange 引用
grep -r "xrange" pyflux/ --include="*.py" && echo "FAIL: xrange found" || echo "PASS: no xrange"

# 检查无 Python 2 版本检查
grep -r "sys.version_info < (3,)" pyflux/ --include="*.py" && echo "FAIL: Python 2 code found" || echo "PASS: Python 2 code removed"

# 运行核心测试
pytest pyflux/gas/tests/gas_tests_normal.py -v
pytest pyflux/arma/tests/test_arima_normal.py -v
```

### 测试结果记录
```
日期: 2026-04-22
xrange 检查: PASS
Python 2 代码检查: PASS
GAS 测试: 24/24 PASS
ARIMA 测试: 17/17 PASS
错误信息: 无
```

---

## 阶段 3: NumPy 弃用类型修复

### 目标
替换 `np.float` 为 `np.float64`

### 检查清单
- [x] 替换所有 `np.float` 为 `np.float64` (17处)
- [x] 验证数据类型转换正确

### 修改文件列表
| 文件 | 行号 | 当前代码 | 修改后 |
|------|------|---------|--------|
| `gas/gas.py` | 60 | `.astype(np.float)` | `.astype(np.float64)` |
| `gas/gasx.py` | 65-66 | `.astype(np.float)` | `.astype(np.float64)` |
| `gas/gasllm.py` | 51 | `.astype(np.float)` | `.astype(np.float64)` |
| `gas/gasllt.py` | 51 | `.astype(np.float)` | `.astype(np.float64)` |
| `gas/gasreg.py` | 56-57 | `.astype(np.float)` | `.astype(np.float64)` |
| `gas/gasrank.py` | 65 | `.astype(np.float)` | `.astype(np.float64)` |
| `arma/arma.py` | 62 | `.astype(np.float)` | `.astype(np.float64)` |
| `arma/arimax.py` | 68-69 | `.astype(np.float)` | `.astype(np.float64)` |
| `arma/nnar.py` | 78 | `.astype(np.float)` | `.astype(np.float64)` |
| `arma/nnarx.py` | 70-71 | `.astype(np.float)` | `.astype(np.float64)` |
| `ssm/llt.py` | 51 | `.astype(np.float)` | `.astype(np.float64)` |
| `ssm/llm.py` | 51 | `.astype(np.float)` | `.astype(np.float64)` |
| `ssm/nllt.py` | 60 | `.astype(np.float)` | `.astype(np.float64)` |
| `ssm/nllm.py` | 60 | `.astype(np.float)` | `.astype(np.float64)` |
| `ssm/dar.py` | 51 | `.astype(np.float)` | `.astype(np.float64)` |

### MVP 测试命令
```bash
# 检查无 np.float 使用 (排除 np.float64)
grep -r "np\.float[^0-9_]" pyflux/ --include="*.py" && echo "FAIL: np.float found" || echo "PASS: no np.float"

# 测试数据类型
python -c "
import numpy as np
from pyflux.gas import GAS
data = np.random.randn(100)
model = GAS(data=data, ar=1, sc=1)
assert model.data.dtype == np.float64
print('PASS: dtype is float64')
"

# 运行测试
pytest pyflux/gas/tests/gas_tests_normal.py -v
pytest pyflux/arma/tests/test_arima_normal.py -v
```

### 测试结果记录
```
日期: 2026-04-22
np.float 检查: PASS
数据类型测试: PASS
GAS 测试: 24/24 PASS
ARIMA 测试: 17/17 PASS
错误信息: 无
```

---

## 阶段 4: Pandas API 变更修复

### 目标
修复已弃用的 Pandas API

### 检查清单
- [x] 替换 `.ix[]` 为 `.iloc[]`
- [x] 替换 `pd.Int64Index` 为 `pd.Index`
- [x] 验证 DataFrame 输入功能

### 修改文件列表
| 文件 | 行号 | 修改内容 |
|------|------|---------|
| `pyflux/data_check.py` | 34 | `data.ix[:,0]` → `data.iloc[:,0]` |
| `pyflux/tsm.py` | 548 | `pd.core.indexes.numeric.Int64Index` → `pd.Index` 类型检查 |
| `pyflux/tsm.py` | 552 | `pd.Int64Index(...)` → `pd.Index(...)` |

### 代码修改示例
```python
# data_check.py 行 34
# Before:
transformed_data = data.ix[:,0].values
# After:
transformed_data = data.iloc[:,0].values

# tsm.py 行 548-552
# Before:
elif isinstance(date_index, pd.core.indexes.numeric.Int64Index):
    for i in range(h):
        new_value = date_index.values[len(date_index.values)-1] + ...
        date_index = pd.Int64Index(np.append(date_index.values,new_value))

# After:
elif isinstance(date_index, pd.Index) and date_index.dtype == 'int64':
    for i in range(h):
        new_value = date_index.values[len(date_index.values)-1] + ...
        date_index = pd.Index(np.append(date_index.values, new_value))
```

### MVP 测试命令
```bash
# 检查无 .ix 使用
grep -r "\.ix\[" pyflux/ --include="*.py" && echo "FAIL: .ix found" || echo "PASS"

# 检查无 Int64Index 使用
grep -r "Int64Index" pyflux/ --include="*.py" && echo "FAIL: Int64Index found" || echo "PASS"

# 测试 DataFrame 输入
python -c "
import pandas as pd
import numpy as np
from pyflux.gas import GAS
df = pd.DataFrame({'value': np.random.randn(100)})
model = GAS(data=df, ar=1, sc=1)
print('PASS: DataFrame input works')
"

# 测试预测功能
python -c "
import numpy as np
from pyflux.gas import GAS
data = np.random.randn(100)
model = GAS(data=data, ar=1, sc=1)
model.fit()
pred = model.predict(h=5)
print('Prediction index type:', type(pred.index))
print('PASS: prediction works')
"
```

### 测试结果记录
```
日期: 2026-04-22
.ix 检查: PASS
Int64Index 检查: PASS
DataFrame 测试: PASS
预测测试: PASS
错误信息: 无
```

---

## 阶段 5: Seaborn API 和 Super 调用更新

### 目标
更新 seaborn distplot 调用和 Python 3 super() 语法

### 检查清单
- [x] 替换 `sns.distplot` 为 `sns.kdeplot` 或 `sns.histplot` (~20处)
- [x] 简化 `super()` 调用 (~15处)
- [x] 验证绘图功能

### Seaborn 修改文件
```
pyflux/latent_variables.py (4处)
pyflux/gas/gas.py, gasx.py, gasreg.py, gasllm.py, gasllt.py
pyflux/garch/garch.py, egarch.py, egarchm.py, egarchmreg.py, segarch.py, segarchm.py, lmegarch.py
pyflux/arma/arma.py, nnar.py, nnarx.py
pyflux/ssm/llt.py, llm.py, dynlin.py
```

### Super 调用修改文件
```
pyflux/gas/*.py (6个文件)
pyflux/arma/*.py (部分文件)
pyflux/families/*.py (多个分布文件)
pyflux/ssm/dar.py
pyflux/inference/bbvi.py
```

### 代码修改示例
```python
# Seaborn 修改
# Before (无直方图):
sns.distplot(data, rug=False, hist=False, label='...')
# After:
sns.kdeplot(data, label='...')

# Before (有直方图):
sns.distplot(data, kde=False, ax=ax)
# After:
sns.histplot(data, ax=ax, kde=False)

# Super 调用修改
# Before:
super(GAS, self).__init__('GAS')
# After:
super().__init__('GAS')
```

### MVP 测试命令
```bash
# 检查无 distplot 使用
grep -r "sns\.distplot" pyflux/ --include="*.py" && echo "FAIL: distplot found" || echo "PASS"

# 测试绘图功能
python -c "
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pyflux.gas import GAS
data = np.random.randn(100)
model = GAS(data=data, ar=1, sc=1)
model.fit()
model.plot_z()
print('PASS: plotting works')
"

# 测试继承
python -c "
from pyflux.gas import GAS, GASX, GASLLEV
from pyflux.families import Normal, t, Poisson
print('PASS: all imports work')
"
```

### 测试结果记录
```
日期: 2026-04-22
distplot 检查: PASS
绘图测试: PASS
导入测试: PASS
错误信息: 无
```

---

## 最终验证

### 完整测试命令
```bash
#!/bin/bash
echo "=== PyFlux Final Validation ==="

echo -e "\n[1/6] Python version"
python --version

echo -e "\n[2/6] Dependency versions"
pip list | grep -E "numpy|pandas|scipy|seaborn|cython"

echo -e "\n[3/6] Build check"
pip install -e . --no-build-isolation

echo -e "\n[4/6] Cython extensions"
python -c "
from pyflux.arma.arma_recursions import arima_recursion
from pyflux.gas.gas_core_recursions import gas_recursion
from pyflux.ssm.kalman import univariate_KFS
print('All Cython extensions OK')
"

echo -e "\n[5/6] Deprecated API check"
echo "Checking for deprecated patterns..."
grep -rE "xrange|np\.float[^0-9_]|Int64Index|\.ix\[|sns\.distplot" pyflux/ --include="*.py" && echo "FAIL: deprecated patterns found" || echo "PASS: no deprecated patterns"

echo -e "\n[6/6] Test suite"
pytest pyflux/ -v --tb=short -x

echo "=== Validation Complete ==="
```

### 验证清单
- [x] 所有 Cython 扩展编译成功
- [x] 无弃用 API 代码残留
- [x] GAS 模块测试全部通过 (24/24)
- [x] ARIMA 模块测试全部通过 (17/17)
- [x] GARCH 模块测试全部通过
- [x] 状态空间模型测试通过

---

## 回滚记录

| 日期 | 阶段 | 回滚原因 | 采取措施 |
|------|------|---------|---------|
| 无 | - | - | - |

---

## 版本发布检查清单

- [x] 所有阶段测试通过
- [x] 更新 `__version__` 为 0.5.0
- [x] 更新 README.md
- [x] 创建 DOCUMENTATION.md 功能文档
- [ ] 创建 Git tag `v0.5.0`
- [ ] 发布到 PyPI (可选)

## 最终测试环境

```
Python: 3.10.20
NumPy: 2.2.6
Pandas: 2.3.3
SciPy: 1.15.3
Cython: 3.2.4
虚拟环境: uv (Python 3.10)
```
