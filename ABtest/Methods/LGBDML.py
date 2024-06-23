import numpy as np
import pandas as pd
from os import cpu_count
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, mean_absolute_percentage_error
import lightgbm as lgb
from statsmodels.api import OLS, add_constant

def lbgdml(data: pd.DataFrame, xColumns, cColumn, yColumn, OneHotColumns=None):

    if OneHotColumns is not None:
        one_hot_data = OneHotEncoder().fit_transform(data[OneHotColumns]).toarray()
        datax = np.hstack(data.loc[:, (~data.columns.isin(OneHotColumns))&(data.columns.isin(xColumns))].values, one_hot_data)
    else:
        datax = data[xColumns].values

    data_all = np.hstack(data[cColumn].values, data[yColumn].values, datax)

    doge = np.random.choice(np.arange(data_all.shape[0]), data_all.shape[0]//2, replace=False)
    data1 = data_all[doge]
    data2 = data_all[~doge]

    model1 = lgb.LGBMClassifier(boosting_type='goss', n_jobs=cpu_count())
    model1.fit(data1[:, 2:], data1[:, 1])
    resid1 = model1.predict_proba(data1[:, 2:])[:, 1]
    resid1[data1[:, 1] == 1] = 1 - resid1[data1[:, 1] == 1]

    model2 = lgb.LGBMClassifier(boosting_type='goss', n_jobs=cpu_count())
    model2.fit(data2[:, 2:], data2[:, 1])
    resid2 = model2.predict_proba(data2[:, 2:])[:, 1]
    resid2[data2[:, 1] == 1] = 1 - resid2[data2[:, 1] == 1]

    model3 = lgb.LGBMRegressor(boosting_type='goss', n_jobs=cpu_count())
    model3.fit(data1[:, 2:], data1[:, 0])
    resid3 = data1[:, 0] - model3.predict(data1[:, 2:])

    model4 = lgb.LGBMRegressor(boosting_type='goss', n_jobs=cpu_count())
    model4.fit(data2[:, 2:], data2[:, 0])
    resid4 = data2[:, 0] - model4.predict(data2[:, 2:])

    print(OLS(resid3, add_constant(resid1)).fit(cov_type='hc0').summary())
    print(OLS(resid4, add_constant(resid2)).fit(cov_type='hc0').summary())