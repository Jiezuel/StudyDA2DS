# sklearn 机器学习笔记



## 聚类-无监督
### K-means
$$ l(\theta)=\sum_{i=1}^m \log\sum_{z=1}^k p(x,z;\theta) \\
=\sum_{i=1}^m \log\sum_{z^{(i)}} \\$$
### Mean-shift
$$暂无公式，摆烂了$$


## 回归
### `Linear Regression` 线性回归
#### 模型
$$ y=\alpha+\beta x+\epsilon $$
```py
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = model.score(y_test, y_pred)
```
### `TheilSen Regressor` 泰尔森回归
泰尔森回归是一种无参数方法，因此无需对数据分布做出假设。
在泰尔森回归中，$\beta$是斜率的中值，因此该模型对异常点有很强的稳健性。
$$ \beta=Median\lbrace \frac{y_i-y_j}{x_i-x_j}:x_i \neq x_j, i<j=1,\ldots,n \rbrace $$
在单变量回归中，泰尔森回归最多允许29.3%的outliers，超出这个范围模型的准确性会大大降低。
```py
from sklearn.linear_model import TheilSenRegressor
```
### `RANSAC Regressor` 随机采样一致性回归
#### RANSAC算法的步骤
- 随机选着估计模型参数所需的最少样本点
- 估计模型参数
- 在误差范围内，将适合该模型的点标为内点
- 重复前面的步骤直到内点数量站样本的比例达到事先设定的阈值
- 基于内点样本拟合回归模型
```py
from sklearn.linear_model import RANSACRegressor
```
### `Huber Regressor`Huber回归
`Huber Regressor`使用Huber损失来优化模型参数，其目标函数如下：
$$ \displaystyle\min_{w,\sigma}\sum_{i=1}^n\left(\sigma+H_\epsilon \left(\frac{X_iw-y_i}{\sigma} \right)\sigma \right)+\alpha||w||_2^2$$
其中：
$$ H_\epsilon(z)=\begin{cases}
    z^2, \qquad\qquad &if|z|<\epsilon\\
    2\epsilon|z|-\epsilon^2 &otherwise \\
\end{cases} \\
z=\frac{X_iw-y_i}{\sigma} $$
其中$\epsilon$为超参数，用来控制MSE和MAE的比例，一般将$\epsilon$设置为1.35；$\alpha$为正则项超参数

```py
from sklearn.linear_model import RANSACRegressor
```
### `Logistic Regressor`逻辑回归
#### 推导
暂时为空
```py
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 500000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
```
### `KNN Regression`KNN回归
```py
from sklearn.neighbors import KNeighborsRegressor
```
### `BayesianRidge`贝叶斯回归
```py
from sklearn.linear_model import BayesianRidge
```

## 分类-有监督
### `Gaussian Process Classifier` 高斯分类器
```py
from sklearn.gaussian_process import GaussianProcessClassifier
```
### `Support Vector Machine`支持向量机
```py
from sklearn.svm import SVC
```
### `Nu Support Vector Classification`
```py
from sklearn.svm import NuSVC
```
### `Naive Bayes Algorithm`朴素贝叶斯
```py
from sklearn.naive_bayes import GaussianNB
```
### `KNN`KNN分类
```py
from sklearn.neighbors import KNeighborsClassifier
```
### `Perceptron`感知器
Perceptron相当于一个二分类单神经网络，它通过超平面将数据分为正负两类。
其模型如下：
$$ f(x)=\sum_{i=1}^Nw_i^Tx_i+b$$
设置阈值$\theta$后表达如下：
$$ y=\begin{cases}
    1\qquad f(x)>\theta \\
    0\qquad f(x)<\theta
\end{cases}$$
损失函数如下：
$$ L(w,b)=-\sum_{x_i\in M}y_i(w^Tx_i+b)$$
```PY
from sklearn.linear_model import Perceptron
```
### `Decision Tree`
```py
from sklearn.tree import DecisionTreeClassfier
```
#### **概述**
决策树方法最早产生于上世纪60年代，到70年代末。由J Ross Quinlan提出了[ID3算法](https://link.springer.com/content/pdf/10.1007/BF00116251.pdf)，此算法的目的在于减少树的深度，但是忽略了叶子数目的研究。C4.5算法在ID3算法的基础上进行了改进，对于预测变量的缺值处理、剪枝技术、派生规则等方面作了较大改进，既适合于分类问题，又适合于回归问题. \
[Incremental Induction of Decision Trees](https://people.cs.umass.edu/~utgoff/papers/mlj-id5r.pdf)
#### **ID3算法**
信息熵
$$ P(X=x_i)=p_i, i=1, 2, ..., n$$
$$ H(X)=-\sum_{i=1}^np_i\log p_i$$
条件熵
$$
\begin{aligned}
H(Y|X)&=\sum_{i=1}^np_iH(Y|X=x_i) \\
&=\sum_{i=1}^np_iH(Y_i) \\
&=\sum_{i=1}^np_i(-\sum_{j=1}^np_j\log p_j)
\end{aligned} 
$$
信息增益 \
特征A对数据集D的信息增益定义Gain如下
$$ g(D,A)=H(D)-H(D|A)$$

#### **ID4.5算法**
[reference](https://zhuanlan.zhihu.com/p/85731206) \
C4.5 相对于 ID3 的缺点对应有以下改进方式:
- 引入悲观剪枝策略进行后剪枝
- 引入信息增益率作为划分标准，防止出现特征/属性的值越多，信息增益越大的情况
- 将连续特征离散化，假设n个样本的连续特征A有m个取值，C4.5将其排序并取相邻两样本值的平均数共m-1个划分点，分别计算以该划分点作为二元分类点时的信息增益，并选择信息增益最大的点作为该连续特征的二元离散分类点

信息增益率定义如下：
$$ Gain_{ratio}(D,A)=\frac{g(D,A)}{H(A)}$$
这里需要注意，信息增益率对可取值较少的特征有所偏好（分母越小，整体越大），因此 C4.5 并不是直接用增益率最大的特征进行划分，而是使用一个**启发式方法**：先从候选划分特征中找到信息增益高于平均值的特征，再从中选择增益率最高的。

剪枝策略
- **预剪枝** 样本低于某一阈值/所有节点特征都已分裂/划分后准确率降低
- **后剪枝** 用递归的方式从低往上针对每一个非叶子节点，评估用一个最佳叶子节点去代替这课子树是否有益。如果剪枝后与剪枝前相比其错误率是保持或者下降，则这棵子树就可以被替换掉。

缺点
- <font color=Blue>剪枝策略可以再优化</font>
- 使用的是多叉树，二叉树效率更高
- 只能用于分类
- 使用熵模型，涉及大量对数运算
- 对连续值离散化涉及排序运算
- 排序运算需要将所有涉及的数据纳入内存中，训练集大时无法运行

#### **CART算法**
改进
- 使用二叉树
- 可以进行回归运算
- 使用Gini系数，减少对数运算
- 使用<font color=Blue>代理测试</font>来估计缺失值
- 使用基于代价复杂度的剪枝策略

Gini系数--不纯度，基尼系数越小，不纯度越小，特征越好
$$ Gini(d)=\sum_{k=1}^K\frac{\vert C_k \vert}{\vert D \vert}(1-\frac{\vert C_k \vert}{\vert D \vert}) \\
=1-\sum_{k=1}^K(\frac{\vert C_k \vert}{\vert D \vert})^2$$

**常用的不纯度函数**定义不纯度函数为$H(D)$ D为某节点的切分变量(特征)
|方法|公式|适用类型|
|---|---|---|
|Gnin系数|$H(D)=\sum_{k=1}^K\frac{\vert C_k \vert}{\vert D \vert}(1-\frac{\vert C_k \vert}{\vert D \vert})$|分类|
|信息熵|$H(D)=-\sum_{k=1}^K\frac{\vert C_k \vert}{\vert D \vert}\log(\frac{\vert C_k \vert}{\vert D \vert})$|分类|
|平方平均误差|$H(D)=\frac{1}{N_c}\sum_{k\in{N_c}}\sum_{i\in C_k}(y_i-\overline{y}_k)^2$|回归|
|绝对平均误差|$H(D)=\frac{1}{N_c}\sum_{k\in{N_c}}\sum_{i\in C_k}\vert y_i-\overline{y}_k\vert$|回归|




**基于代价复杂度的剪枝策略？？？**

**类别不平衡问题的解决**
当且仅当：
$$ \frac{N_1(node)}{N_1(root)} > \frac{N_0(node)}{N_0(root)}$$
节点node被划分为1类

**CART回归树**
连续值的处理？？？

### `Random Forest`随机森林
`RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)`
```py
from sklearn.ensemble import RandomForestClassifier
```
[Random Forest](https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf) 由Leo Breiman在2001年提出

**何为随机**
- 随机样本
- 随机特征
- 随机树



### 提升树
