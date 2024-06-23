##


## 样本量


|Situation|Sample Size to Estimate Confidence Interval|Sample Size to Conduct Test of Hypothesis|
|:--:|:--:|:--:|
|Continuous Outcome <br>One Sample<br>CI for $\mu,H_0:\mu=\mu_0$|$n=(\frac{Z_{\alpha}}{E})^2$|$n=(\frac{Z_{1-\alpha/2}+Z_{1-\beta}}{ES})^2$ <br> $ES=\frac{\|\mu-\mu_0\|}{\sigma}$|
|Continuous Outcome <br>Two Independent Samples<br>CI for $(\mu_1-\mu_2),H_0:\mu_1=\mu_2$|$n=2(\frac{Z_{\alpha}}{E})^2$|$n=2(\frac{Z_{1-\alpha/2}+Z_{1-\beta}}{ES})^2$ <br> $ES=\frac{\|\mu_1-\mu_2\|}{\sigma}$|
|Continuous Outcome <br>Two Matched Samples<br>CI for $\mu_d,H_0:\mu_d=0$|$n=(\frac{Z_{\alpha d}}{E})^2$|$n=(\frac{Z_{1-\alpha/2}+Z_{1-\beta}}{ES})^2$ <br> $ES=\frac{\mu_d}{\sigma_d}$|
|Dichotomous Outcome <br>One Sample<br>CI for $p,H_0:p=p_0$|$n=p(1-p)(\frac{Z}{E})^2$|$n=(\frac{Z_{1-\alpha/2}+Z_{1-\beta}}{ES})^2$ <br> $ES=\frac{p_1-p_0}{\sqrt{p_1(1-p_1)}}$|
|Dichotomous Outcome <br>Two Independent Samples<br>CI for $(p_1-p_2),H_0:p_1=p_2$|$n=\lbrace p_1(1-p_1)+p_2(1-p_2)\rbrace (\frac{Z}{E})^2$|$n=2(\frac{Z_{1-\alpha/2}+Z_{1-\beta}}{ES})^2$ <br> $ES=\frac{\|p_1-p2\|}{\sqrt{p(1-p)}}$|


``` py
import math
from statsmodels.stats.power import NormalIndPower

ztest = NormalIndPower()
es = 0.001/math.sqrt(0.01*(1-0.01))
sample_size = ztest.solve_power(effect_size=es, nobs1=None, alpha=0.05, power=0.8, ratio=1, alternative= 'two-sided')
sample_size
```
## 分组
### 事后寻找对照组
#### 欧式距离法
``` py
def search_sample(data, train_data, seed=20, weight=None):
    a = train_data.loc[data.distributed==1]
    b = train_data.loc[data.distributed==0].copy()
    control = []
    control_pool = []
    print(len(a))
    not_match = []
    for i in range(len(a)):
        a_series = a.iloc[i, :]
        dist = b.sub(a_series, axis='columns')
        dist = dist.apply(lambda x: x*x)
        if weight is not None:
            assert len(weight)==dist.shape[1]
            dist = dist.mul(weight, axis='columns')
        dist = dist.sum(axis=1).sort_values()
        idxs = dist.head(seed).index.to_list()
        j = 0
        for idx in idxs:
            if idx not in control:
                control.append(idx)
                break
            else:
                j +=1
        idxs.insert(0, a_series.name)
        if j==seed:
            print(idxs)
            not_match.append(a_series.name)
        control_pool.append(idxs)

    return control, control_pool, not_match
```

#### PSM得分
``` py
import statsmodels.api as sm
import pandas as pd
data = pd.DataFrame()
x_columns = data.drop('T', axis=1).columns
formula = "T ~ " + '+'.join(x_columns)
model = sm.Logit.from_formula(formula, data = data)
reg = model.fit()
X = data[x_columns]
data['psm_score'] = reg.predict(X)
```

#### 寻找最优PSM模型
``` py
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
import patsy
import sys
def train_psm_2(data, nmodels=50, n=None):
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
            print('Average Accuracy: {:.2%}, Recall: {}'.format(ab_score, recall))
            i+=1
        except Exception as e:
            errors += 1
            print('Error: {}'.format(e))
  
    return models, model_accuracy

```


### 事前分组方法
#### 大样本随机分组
#### 聚类分层抽样
#### PSM匹配
#### CUPED
#### K-FOLD ML 效应得分

### 稀疏以及不稳定数据匹配
- SCM