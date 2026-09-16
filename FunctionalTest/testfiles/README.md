# Functional tests: Sections 1–7

在仓库根目录执行（先激活你的 Python 环境）：

```bash
python -m pip install -r FunctionalTest/testfiles/requirements-test.txt
python -m pytest -c FunctionalTest/testfiles/pytest.ini -v
```

也可以进入本目录后执行 `python -m pytest -v`。

输入输出复核说明见 [INPUT_OUTPUT_CHECK.md](INPUT_OUTPUT_CHECK.md)。

`lib/corr_and_covar.py` 的四个函数接收二维数值数据：行是观测，列是变量，
缺失值用 `np.nan` 表示；返回 NumPy 矩阵。原来的 `(x, y)` 接口改为 `(data)`，
以便整行删除能够检查所有变量的缺失值。CSV 读取由测试负责。

| Tests.xlsx 编号 | 计算方式 | 输入 | 标准输出 |
| --- | --- | --- | --- |
| 1.1 | 整行删除缺失值，样本协方差 | data/test1.csv | data/testout_1.1.csv |
| 1.2 | 整行删除缺失值，Pearson 相关系数 | data/test1.csv | data/testout_1.2.csv |
| 1.3 | 成对删除缺失值，样本协方差 | data/test1.csv | data/testout_1.3.csv |
| 1.4 | 成对删除缺失值，Pearson 相关系数 | data/test1.csv | data/testout_1.4.csv |
| 2.1 | EW 协方差，λ=0.97 | data/test2.csv | data/testout_2.1.csv |
| 2.2 | EW 相关系数，λ=0.94 | data/test2.csv | data/testout_2.2.csv |
| 2.3 | EW 方差 λ=0.97，EW 相关系数 λ=0.94 | data/test2.csv | data/testout_2.3.csv |
| 3.1 | near_psd 修正协方差 | data/testout_1.3.csv | data/testout_3.1.csv |
| 3.2 | near_psd 修正相关系数 | data/testout_1.4.csv | data/testout_3.2.csv |
| 3.3 | Higham 修正协方差 | data/testout_1.3.csv | data/testout_3.3.csv |
| 3.4 | Higham 修正相关系数 | data/testout_1.4.csv | data/testout_3.4.csv |
| 4.1 | PSD Cholesky 分解 | data/testout_3.1.csv | data/testout_4.1.csv |
| 5.1 | 正定矩阵正态模拟 | data/test5_1.csv | data/testout_5.1.csv |
| 5.2 | 半正定矩阵正态模拟 | data/test5_2.csv | data/testout_5.2.csv |
| 5.3 | near_psd 修正后模拟 | data/test5_3.csv | data/testout_5.3.csv |
| 5.4 | Higham 修正后模拟 | data/test5_3.csv | data/testout_5.4.csv |
| 5.5 | PCA 模拟，保留 99% 方差 | data/test5_2.csv | data/testout_5.5.csv |
| 6.1 | 算术收益率 | data/test6.csv | data/testout6_1.csv |
| 6.2 | 对数收益率 | data/test6.csv | data/testout6_2.csv |
| 7.1 | 正态分布拟合 | data/test7_1.csv | data/testout7_1.csv |
| 7.2 | 广义 t 分布最大似然拟合 | data/test7_2.csv | data/testout7_2.csv |
| 7.3 | t 分布误差回归 | data/test7_3.csv | data/testout7_3.csv |
| 7.4 | t 分布拟合的 AICc | data/test7_2.csv | data/testout7_4.csv |
| 7.5 | NIG 矩估计 | data/test7_5.csv | data/testout7_5.csv |
| 7.6 | NIG 最大似然拟合 | data/test7_5.csv | data/testout7_6.csv |

第 7 部分的函数位于 `lib/distribution_fitting.py`，对应测试为
`tests/test_distribution_fitting.py`，需要 SciPy。拟合函数返回参数字典：

| 编号 | 函数 | 返回值 |
| --- | --- | --- |
| 7.1 | fit_normal(data) | mu, sigma（样本标准差） |
| 7.2 | fit_general_t(data) | mu, sigma（尺度参数）, nu |
| 7.3 | fit_regression_t(x, y) | mu=0, sigma, nu, Alpha, B1, B2, ... |
| 7.4 | aicc(log_likelihood, n, k) | AICc 数值 |
| 7.5 | fit_nig_moments(data) | mu, alpha, beta, delta |
| 7.6 | fit_nig_mle(data) | mu, alpha, beta, delta |

7.3 的 x 是不含截距列的二维自变量矩阵，y 是一维因变量；函数同时拟合截距、
回归系数和 t 误差分布。7.4 使用 7.2 的拟合参数计算总对数似然，再传入
`aicc(log_likelihood, len(data), 3)`，其中 3 是估计参数的数量。
7.5 使用样本方差和未经偏差修正的偏度、超额峰度，与标准输出约定一致；
无法产生有限 NIG 参数的样本矩会抛出异常。

7.6 将 SciPy 的 `(a, b, loc, scale)` 转为 `(mu, alpha, beta, delta)`，
转换关系见 [SciPy NIG 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norminvgauss.html)。
7.1、7.5 使用 `rtol=1e-10` 对比；数值优化会有末位差异，7.2、7.6 使用
`rtol=1e-4`，7.3 使用 `rtol=1e-5`，7.4 使用 `rtol=1e-8`。
测试还检查 NIG 的矩还原、似然改进及无效输入，不只是比较参数。

第 4–6 部分按函数文件和测试文件对应组织：

| 函数文件 | 主要函数 | 测试文件 |
| --- | --- | --- |
| lib/chol_psd.py | chol_psd(matrix) | tests/test_chol_psd.py |
| lib/simulations.py | simulate_normal / simulate_pca | tests/test_simulations.py |
| lib/returns_conversion.py | return_calculate | tests/test_returns_conversion.py |

`chol_psd` 返回下三角矩阵 L，使 `L @ L.T` 还原输入，支持奇异半正定矩阵。
4.1 的末尾零主元可能存在浮点误差，因此标准答案对比使用 `atol=1e-8`，
同时用 `atol=1e-12` 检查矩阵重建。

`simulate_normal(100000, covariance, fix_method=None, seed=545)` 返回零均值
正态样本，每行一次模拟。5.3、5.4 分别传入 `near_psd`、`higham_nearest_psd`。
`simulate_pca(covariance, 100000, explained=0.99, seed=545)` 保留累计方差达到
99% 所需的最少主成分。Python 和 Julia 的随机数不同，第 5 部分不要求逐元素
精确相等：测试按正态样本协方差的标准误差设置六倍误差范围，分别检查理论值、
标准 CSV 和模拟结果；两份模拟结果比较时计入双方的抽样误差。

`return_calculate(prices, method="arithmetic", date_column="Date")` 接收 pandas
DataFrame；`method="log"` 计算对数收益率。价格须为正且有限，按输入行序计算，
输出保留列顺序及每个收益区间的结束日期。Excel 中第 6 部分的文件名与磁盘上
略有不同，测试使用实际存在的 `testout6_1.csv` 和 `testout6_2.csv`。
7.1–7.4 也有相同的命名差异，测试使用实际存在的 `testout7_1.csv` 至
`testout7_4.csv`。所有标准数据均保留原样。

`lib/fix_nonpsd_corr_covar.py` 实现 `near_psd(matrix)` 和
`higham_nearest_psd(matrix)`，分别用于协方差矩阵和相关系数矩阵。
输入必须是有限、对称、对角线为正的方阵。协方差先转换到相关系数尺度，
修正后恢复原方差。Higham 在相关系数尺度上判断收敛，默认容差为 `1e-9`，
最多迭代 1000 次；未收敛会抛出异常。结果可能存在容差范围内的微小负特征值。

`lib/expo_weighted_corr_and_covar.py` 实现 2.1–2.3：`ew_covariance`、
`ew_correlation` 和 `ew_covariance_combined`。输入按时间从旧到新排列，
最后一行权重最大。权重归一化后使用加权均值，不额外做 `n - 1` 修正。
这些函数要求至少两行有限数值，不接受缺失值。2.3 的参数按 Excel 和标准
输出实现；`test_setup.jl` 中 2.3 的文字注释把两个 λ 写反了。

第 1–3、6 部分逐元素比较标准输出（`rtol=1e-10, atol=1e-12`），并检查缺失值策略、
观测不足、常数列和无共同观测等边界情况。样本协方差使用 `n - 1`；
相关系数在每一对共同有效的观测上计算。少于两个有效观测返回 NaN；
相关系数涉及常数列时也返回 NaN。

Tests.xlsx 和标准答案 CSV 只用于对照，不会被测试修改。
