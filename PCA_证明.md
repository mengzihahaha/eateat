# PCA (Principal Component Analysis) 推导证明

## 1. 问题设定

给定 $n$ 个样本点 $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n \in \mathbb{R}^d$，已中心化（$\bar{\mathbf{x}} = \mathbf{0}$）。

目标：找一个 $k$ 维子空间（$k < d$），使投影后的数据保留最大信息量（方差）。

---

## 2. 第一主成分的推导

### 2.1 目标函数

寻找单位向量 $\mathbf{u}_1$（$\|\mathbf{u}_1\| = 1$），使投影 $\mathbf{x}_i^\top \mathbf{u}_1$ 的样本方差最大：

$$\max_{\mathbf{u}_1} \frac{1}{n} \sum_{i=1}^{n} (\mathbf{x}_i^\top \mathbf{u}_1)^2 = \mathbf{u}_1^\top \mathbf{S} \mathbf{u}_1$$

其中协方差矩阵 $\mathbf{S} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i \mathbf{x}_i^\top = \frac{1}{n} \mathbf{X}^\top \mathbf{X}$。

### 2.2 拉格朗日乘数法

构造拉格朗日函数：

$$\mathcal{L}(\mathbf{u}_1, \lambda) = \mathbf{u}_1^\top \mathbf{S} \mathbf{u}_1 - \lambda (\mathbf{u}_1^\top \mathbf{u}_1 - 1)$$

对 $\mathbf{u}_1$ 求导并令其为零：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_1} = 2\mathbf{S}\mathbf{u}_1 - 2\lambda \mathbf{u}_1 = \mathbf{0}$$

$$\Rightarrow \mathbf{S}\mathbf{u}_1 = \lambda \mathbf{u}_1$$

这说明 $\mathbf{u}_1$ 是 $\mathbf{S}$ 的特征向量，$\lambda$ 是对应特征值。

### 2.3 最优性证明

将特征值方程代入目标函数：

$$\mathbf{u}_1^\top \mathbf{S} \mathbf{u}_1 = \mathbf{u}_1^\top (\lambda \mathbf{u}_1) = \lambda \mathbf{u}_1^\top \mathbf{u}_1 = \lambda$$

因此方差等于对应特征值 $\lambda$。为使方差最大，$\mathbf{u}_1$ 应取 $\mathbf{S}$ 的**最大特征值** $\lambda_1$ 对应的特征向量。 $\blacksquare$

---

## 3. 第 $k$ 个主成分的推导

### 3.1 带正交约束的优化

第 $k$ 个主成分 $\mathbf{u}_k$ 需满足：

- $\|\mathbf{u}_k\| = 1$
- $\mathbf{u}_k^\top \mathbf{u}_j = 0, \quad \forall j < k$（与前 $k-1$ 个主成分正交）

$$\max_{\mathbf{u}_k} \mathbf{u}_k^\top \mathbf{S} \mathbf{u}_k$$

### 3.2 拉格朗日函数

$$\mathcal{L}(\mathbf{u}_k, \lambda, \boldsymbol{\mu}) = \mathbf{u}_k^\top \mathbf{S} \mathbf{u}_k - \lambda(\mathbf{u}_k^\top \mathbf{u}_k - 1) - \sum_{j=1}^{k-1} \mu_j \mathbf{u}_k^\top \mathbf{u}_j$$

对 $\mathbf{u}_k$ 求导：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_k} = 2\mathbf{S}\mathbf{u}_k - 2\lambda \mathbf{u}_k - \sum_{j=1}^{k-1} \mu_j \mathbf{u}_j = \mathbf{0}$$

左乘 $\mathbf{u}_j^\top$（$j < k$）：

$$2\mathbf{u}_j^\top \mathbf{S} \mathbf{u}_k - \mu_j = 0$$

由于 $\mathbf{S}$ 对称，$\mathbf{u}_j^\top \mathbf{S} \mathbf{u}_k = (\mathbf{S}\mathbf{u}_j)^\top \mathbf{u}_k = \lambda_j \mathbf{u}_j^\top \mathbf{u}_k = 0$。

因此 $\mu_j = 0$，退化为：

$$\mathbf{S}\mathbf{u}_k = \lambda \mathbf{u}_k$$

同样得到特征值方程。$\mathbf{u}_k$ 取剩余特征向量中对应最大特征值的那个。 $\blacksquare$

---

## 4. 整体最优性（全局最优）

### 4.1 投影矩阵形式

设 $\mathbf{U} = [\mathbf{u}_1, \dots, \mathbf{u}_k] \in \mathbb{R}^{d \times k}$，$\mathbf{U}^\top \mathbf{U} = \mathbf{I}_k$。

总投影方差为：

$$J(\mathbf{U}) = \sum_{j=1}^{k} \mathbf{u}_j^\top \mathbf{S} \mathbf{u}_j = \mathrm{tr}(\mathbf{U}^\top \mathbf{S} \mathbf{U})$$

### 4.2 Ky Fan 定理

> **定理 (Ky Fan)：** 对于对称矩阵 $\mathbf{S}$，其特征值 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$，则
> $$\max_{\mathbf{U}^\top \mathbf{U} = \mathbf{I}_k} \mathrm{tr}(\mathbf{U}^\top \mathbf{S} \mathbf{U}) = \sum_{j=1}^{k} \lambda_j$$
> 当且仅当 $\mathbf{U}$ 的列空间为前 $k$ 个最大特征值对应的特征空间时取等。

**证明思路：**

设 $\mathbf{S}$ 的谱分解为 $\mathbf{S} = \sum_{i=1}^{d} \lambda_i \mathbf{v}_i \mathbf{v}_i^\top$。

令 $\alpha_{ij} = (\mathbf{u}_j^\top \mathbf{v}_i)^2$，则 $\alpha_{ij} \geq 0$，且：

- $\sum_{i=1}^{d} \alpha_{ij} = \mathbf{u}_j^\top \mathbf{u}_j = 1$
- $\sum_{j=1}^{k} \alpha_{ij} = \sum_{j=1}^{k} (\mathbf{v}_i^\top \mathbf{u}_j)^2 \leq \|\mathbf{v}_i\|^2 = 1$

$$J(\mathbf{U}) = \sum_{j=1}^{k} \sum_{i=1}^{d} \lambda_i \alpha_{ij} \leq \sum_{j=1}^{k} \lambda_j \sum_{i=1}^{d} \alpha_{ij} \leq \sum_{j=1}^{k} \lambda_j$$

当 $\mathbf{u}_j = \mathbf{v}_j$（$j = 1, \dots, k$）时取等号。 $\blacksquare$

---

## 5. 最小重构误差等价性

### 5.1 重构误差

用 $k$ 维子空间重构数据，重构误差为：

$$E = \sum_{i=1}^{n} \left\| \mathbf{x}_i - \sum_{j=1}^{k} (\mathbf{x}_i^\top \mathbf{u}_j) \mathbf{u}_j \right\|^2$$

### 5.2 等价性证明

展开重构误差：

$$E = \sum_{i=1}^{n} \|\mathbf{x}_i\|^2 - \sum_{i=1}^{n} \sum_{j=1}^{k} (\mathbf{x}_i^\top \mathbf{u}_j)^2$$

第一项为常数，因此：

$$\min E \iff \max \sum_{j=1}^{k} \mathbf{u}_j^\top \mathbf{S} \mathbf{u}_j$$

即**最大方差**与**最小重构误差**等价。 $\blacksquare$

---

## 6. 结论

| 性质 | 结果 |
|---|---|
| 第 $j$ 个主成分方向 | $\mathbf{S}$ 的第 $j$ 大特征值对应的特征向量 |
| 第 $j$ 主成分的方差 | $\lambda_j$ |
| $k$ 个主成分的累计方差贡献率 | $\frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{d} \lambda_j}$ |
| PCA 等价形式 | 最大方差 $\Leftrightarrow$ 最小重构误差 |
| 计算复杂度 | $O(d^2 n + d^3)$（标准 SVD） |
