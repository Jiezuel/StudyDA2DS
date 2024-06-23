# 统计检验
## Part1 scipy
### 正态性检验
$H_0：数据符合正态分布$ \
`rvs: str, array_like, or callable`样本数据 \
`cdf: str, array_like or callable`样本分布
- `scipy.stats.shapiro(rvs)`
- `scipy.stats.kstest(rvs, cdf)`
- Jarque-Bera检验
  - `scipy.stats.jarque_bera(rvs)`
    $$ JB=n(\frac{S^2}{6} + \frac{(K-3)^2}{24})$$
  - n样本数量
  - S偏度，K峰度
### T检验
#### 单样本T检验
`scipy.stats.ttest_1samp(array, mean, axis=0)`检验样本均值是否与某一值相等
#### 双样本T检验
`scipy.stats.ttest_ind(array1, array2)`检验两样本均值是否相等
#### 配对样本t检验（检验两个样本所代表的总体的均值是否相等）
### 双样本方差齐性检验
$H_0：两数据具有方差齐性$ \
`scipy.stats.levene(array1, array2)`检验两样本方差分布是否一致
### 相关性检验
`scipy.stats.pearsonr(array1, array2)`皮尔森相关系数检验 \
`scipy.stats.spearmanr(array1, array2)`斯皮尔曼等级相关系数(Spearman’s correlation coefficient for ranked data ) 不考虑变量值大小，只考虑变量排名，常用来衡量类型变量的相关性
### 相关性分析

<table>
    <tr>
        <th colspan="2">x\y</th>
        <th>二分类</th>
        <th>连续</th>
    </tr>
    <tr>
        <td rowspan="3">单变量</td>
        <td>二分类</td>
        <td>列联表分析|卡方检验</td>
        <td>双样本t检验</td>
    </tr>
    <tr>
        <td>多分类</td>
        <td>列联表分析|卡方检验</td>
        <td>单因素方差分析</td>
    </tr>
    <tr>
        <td>连续</td>
        <td>双样本t检验</td>
        <td>相关分析</td>
    </tr>   
    <tr>
        <td rowspan="2">多变量</td>
        <td>分类</td>
        <td>逻辑回归</td>
        <td>多因素方差分析|线性回归</td>
    </tr>   
    <tr>
        <td>连续</td>
        <td>逻辑回归</td>
        <td>线性回归</td>    
    </tr>      
</table>

#### 连续变量相关系数R^2的计算
$$
r = \frac{cov(x,y)}{\sigma_x\sigma_y}=\frac{\sum_{i=1}^n(x_i-\overline{x})(y_i-\overline{y})}{\sqrt{\sum_{i=1}^n(x_i-\overline{x})^2\sum_{i=1}^n(y_i-\overline{y})^2}}
$$

#### 相关系数**T**统计量的计算
$$ t=\frac{r\sqrt{n-2}}{\sqrt{1-r^2}}$$

### 确定两份数据同分布
#### **KS检验(连续变量)**--拟合优度检验
通过比较两份数据的累积概率分布的最大距离来衡量数据的接近程度 \
前提：满足**正态性**或**方差齐性**
```py
from scipy.stats import ks_2samp
ks_2samp(array1, array2).pvalue
```
#### **非参数检验**
- **Wilcoxon符号秩检验**`scipy.stats.wilcoxon`
  - 样本长度一致
  - 样本总体的偏差服从正态分布
  - 样本量大于20
- **Kruskal-Wallis H检验**`scipy.stats.kruskal`
  - 假设两个分布的样本中位数相等，用于检验样本是否来源于相同的分布
  - 样本量大于5
  - 两组数据长度可以不一致
- **Mann-Whitney秩检验**`scipy.stats.mannwhitneyu`
  - 假设两个样本分别来自除了总体均值以外完全相同的两个总体，目的是检验这两个总体的均值是否有显著的差别
  - 样本量大于20
  - 两组数据长度可以不一致
#### **KL散度(连续变量)**
KL 散度是一种衡量两个概率分布的匹配程度的指标，两个分布差异越大，KL散度越大，但是该方法是不对称的，查看的测试集对训练集的匹配程度
#### **构建二分类模型**
通过构建二分类模型，如果分类效果良好，说明两个样本差异较大。


## Part2 statsmodels
### 多元线性回归检验
``` py
import statsmodels.api as sm
model = sm.OLS(data).fit()
print(model.summary())
```

### 结构性异变检验
``` py
from sklearn.linear_model import LinearRegression as Lr
def calculate_rss(x, y):
    # if x.shape[0] == 1:
    #     x = x.reshape(-1, 1)
    model = Lr().fit(x, y)
    y_hat = pd.Series(model.predict(x))
    residuals = y.reset_index(drop=True) - y_hat
    return (residuals**2).sum()

def chow_statistic(s_c, s_1, s_2, k, n_1, n_2):
    numerator = (s_c - s_1 - s_2) / k 
    denominator = (s_1 + s_2) / (n_1 + n_2 - (2 * k))
    return numerator / denominator

def p_value_f(chow_statistic, k, n_1, n_2):
    from scipy.stats import f
    p_value = float(1 - f.cdf(chow_statistic, dfn=k, dfd=((n_1 + n_2) - 2 * k)))
    return p_value

def chow_test(df, x_comlumns, y_column, split_column):
    df1 = df.loc[df[split_column]==0].copy()
    df2 = df.loc[df[split_column]==1].copy()
    x_1, y_1 = df1[x_comlumns], df1[y_column]
    x_2, y_2 = df2[x_comlumns], df2[y_column]
    rss_all = calculate_rss(df[x_comlumns], df[y_column])
    rss_1 = calculate_rss(x_1, y_1)
    rss_2 = calculate_rss(x_2, y_2)
    k = x_1.shape[1] + 1
    n_1 = x_1.shape[0]
    n_2 = x_2.shape[0]
    chow_value = chow_statistic(rss_all, rss_1, rss_2, k, n_1, n_2)
    p_value = p_value_f(chow_value, k, n_1, n_2)
    print('Chow value is {}, p_values is {}'.format(chow_value, p_value))
    return chow_value, p_value
```
### PSM分组
``` py
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
def train_psm(data, nmodels=50, n=None):
    field = train_data.columns.values
    X_field = field[1:]
    Y_field = field[:1]
    formula = '{} ~ {}'.format(Y_field[0], '+'.join(X_field))

    i = 0 
    errors = 0
    model_accuracy = []
    models = []
    while i < nmodels and errors < 5:
        sys.stdout.write('{}_{}_{}'.format("Fitting Models on Balanced Samples", i, nmodels))
        df = data.loc[data.distributed==0].sample(n).append(data.loc[data.distributed==1], ignore_index=True)
        try:
            model = sm.Logit.from_formula(formula, data=df,)
            LR = model.fit(method='bfgs')
            X = df[X_field]
            y_pre = LR.predict(X)
            a = [1.0 if i >= .5 else 0.0 for i in y_pre] 
            b = y_pre
            ab_score = ((a==b).sum() * 1.0 / len(y_pre))
            recall = ((a==b)&(b==1)).sum()
            model_accuracy.append(recall)
            models.append(LR)
            print('___Average Accuracy: {:.2%}, Recall: {}'.format(ab_score, recall))
            i+=1
        except Exception as e:
            errors += 1
            print('_____Error: {}'.format(e))
  
    return models, model_accuracy

```



### Pandas可视化
#### 多维分类散点图
`pandas.plotting.scatter_matrix` 分类颜色需要自己设置
``` py
import pandas as pd
pd.plotting.scatter_matrix(
    data # df,
    c, #color-->scalar or list
    figsize, # list
    diagonal, # str, histohram of each features
    s, # num, size of marker
    marker, # str, type of marker, eg: *, -, + 
)
```






