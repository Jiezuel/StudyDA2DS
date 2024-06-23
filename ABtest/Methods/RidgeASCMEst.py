"""
Created on Jun 9 2022

@author: WanShanLe
"""

import pandas as pd
import numpy as np
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.preprocessing import minmax_scale
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from numpy import array


class ScmEst:
    @staticmethod
    def min_scm(gamma, x_1, x_0, V_x):
        gamma = array(gamma).reshape(-1, 1)
        scm = ((x_1 - x_0.T @ gamma).T @ V_x @ (x_1 - x_0.T @ gamma))[0][0]

        return scm

    def __init__(self, p_data, n_data, disp=0):
        t_all = p_data.shape[0]
        sc_vars = len(n_data)
        V_x = np.diag([1] * t_all)
        x0 = [1 / sc_vars] * sc_vars  # 初始化参数——等权
        # 权重之和为1的约束
        cons = {"type": "eq", "fun": lambda x: sum(x) - 1}
        # 各个权重应该大于等于0的约束
        bnds = eval("(0, None)," * sc_vars)
        res = minimize(
            self.min_scm,
            x0,
            args=(p_data, n_data, V_x),
            method="SLSQP",
            tol=1e-6,
            constraints=cons,
            bounds=bnds,
        )
        self.weight = res.x
        if disp != 0:
            print("scm权重和:", np.round(np.sum(self.weight), 2))
            print("scm优化结果:", res.fun)


class RidgeASCM:
    @staticmethod
    def min_ascm(gamma, x_1, x_0, V_x, para, w_scm):
        gamma = array(gamma).reshape(-1, 1)
        ascm = (
                para * ((x_1 - x_0.T @ gamma).T @ V_x @ (x_1 - x_0.T @ gamma))
                + 0.5 * (gamma - w_scm).T @ (gamma - w_scm)
        )[0][0]
        return ascm

    def __init__(self, p_data, n_data, lambda_ridge, disp=0):
        scm_weight = ScmEst(p_data, n_data).weight.reshape(-1, 1)
        t_all = p_data.shape[0]
        sc_vars = len(n_data)
        V_x = np.diag([1] * t_all)
        x0 = [1 / sc_vars] * sc_vars  # 初始化参数——等权
        # 权重之和为1的约束
        cons = {"type": "eq", "fun": lambda x: sum(x) - 1}
        # 各个权重应该大于等于0的约束
        bnds = eval("(0, None)," * sc_vars)
        para = 1 / (2 * lambda_ridge)
        res = minimize(
            self.min_ascm,
            x0,
            args=(p_data, n_data, V_x, para, scm_weight),
            method="SLSQP",
            tol=1e-6,
            constraints=cons,
            bounds=bnds,
        )
        self.weight = res.x
        if disp != 0:
            print("ridgeASCM权重和:", np.round(np.sum(self.weight), 2))
            print("ridgeASCM优化结果:", res.fun)


class RidgeASCMEst:
    def best_para_fit(self):
        mse_all = [0] * 7
        for i, para_ridge in enumerate(self.lambda_ridge):
            ridge_ascm_w = RidgeASCM(self.p_0, self.n_0, para_ridge).weight.reshape(
                1, -1
            )
            ridge_ascm_n = (ridge_ascm_w @ self.n_0).T
            mse_all[i] = mean_squared_error(self.p_0, ridge_ascm_n)
        self.best_para = self.lambda_ridge[np.argmin(mse_all)]

    def __init__(self, tra_data_0, con_data_0, tra_data_1, con_data_1):
        self.p_0 = tra_data_0
        self.n_0 = con_data_0
        self.p_1 = tra_data_1
        self.n_1 = con_data_1
        self.lambda_ridge = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        self.best_para = 1.0
        self.best_para_fit()
        self.ridge_ascm_w = RidgeASCM(
            self.p_1, self.n_1, self.best_para, disp=1
        ).weight.reshape(1, -1)
        self.ridge_ascm_n = (self.ridge_ascm_w @ self.n_1).T
        print("最优参数:", self.best_para)
        print("MSE:", mean_squared_error(self.p_1, self.ridge_ascm_n))
        print("diff:", (self.p_1 - self.ridge_ascm_n).reshape(-1))


"""
一般来说,在做lowbi的AB测试的时候,如果不摞GMM或者IV以工具变量的方法进行估计,PSM-DID是一个解决内生性问题的良好工具(PSM方法上面给出)
但是对于DID来说由于控制组只能选取一个,具有较大的个人主观性(结果容易操纵)和同质性假设强(要求对照组测试组统治)
The Economic Costs of Conflict: A Case Study of the Basque Country, Abadie, AER, 2003
SCM通过使用对一摞控制组加权组合得到相对更为自然且对照组合成结果客观的方法得到青睐
但对于SCM来说在处理平均处理效应ATE的时候,会有一个很强的数据假设:处理效应明显-即要求处理必要充分对对照组进行干预,否则容易出现系数偏误
增强合成控制方法ASCM在估计的最小化时通过对系数偏差进行修正(增加偏误项的罚项)以达到减少处理效应不够强的时候的系数偏误问题
ASCM的罚项在减少处理效应不够强时强依赖于罚项的选择(否则不受明显影响),且有参数化和非参数化方法.
The Augmented Synthetic Control Method, Eli Ben-Michael, NEBR, 2021
Eli Ben-Michael等人提出了可以用Ridge岭回归对SCM进行增强,称为RidgeASCM
RidgeASCM首先用常规方法估计SCM的权重向量,随后优化带Ridge罚项的得到RidgeASCM的权重向量
在超参数的选择上,则提出了可以通过CV最小化非处理时间段的测试组组和RidgeASCM估计的对照组的MSE



p_0和p_1是测试开始前和开始后的测试组数据
n_0和n_1是测试开始前和开始后的对照组数据
"""
len_t = 10
len_con = 50
p_0 = np.abs(np.random.rand(len_t)).reshape(-1, 1) * 45
p_1 = np.abs(np.random.rand(len_t)).reshape(-1, 1) * 65
n_0 = np.abs(np.random.rand(len_con, len_t)) * 50
n_1 = np.abs(np.random.rand(len_con, len_t)) * 48

result1 = ScmEst(p_0, n_0, disp=1)

result2 = RidgeASCMEst(p_0, n_0, p_1, n_1)

