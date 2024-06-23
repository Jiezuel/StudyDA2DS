# -*- coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class QuantileRegressionForests(RandomForestRegressor):
    def __init__(
        self,
        n_estimators=10,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        iqr_coef=1.2,
        exclude_side='both'
    ):
        super(QuantileRegressionForests, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )
        self.iqr_coef = iqr_coef
        if exclude_side in ['upper_only', 'lower_only']:
            self.exclude_side = exclude_side
        else:
            self.exclude_side = 'both'

    def _predicted_values(self, X):

        predicted_list = list()

        for decision_tree_regressor in self.estimators_:

            predicted_list.append(
                decision_tree_regressor.predict(X)
            )
        predicted_list = np.array(predicted_list).T

        q1 = np.percentile(predicted_list, 25, axis=1)
        q3 = np.percentile(predicted_list, 75, axis=1)
        iqr = self.iqr_coef * (q3 - q1)

        under_limits = q1 - iqr
        upper_limits = q3 + iqr

        if self.exclude_side == 'upper_only':
            ret = np.array([
                np.where(values <= upper, values, np.nan)
                for values, upper
                in zip(predicted_list, upper_limits)
            ])
        elif self.exclude_side == 'lower_only':
            ret = np.array([
                np.where(under <= values, values, np.nan)
                for values, under
                in zip(predicted_list, under_limits)
            ])
        else:
            ret = np.array([
                np.where((under <= values) & (values <= upper), values, np.nan)
                for values, under, upper
                in zip(predicted_list, under_limits, upper_limits)
            ])
        return ret

    def predict(self, X):

        predicted_values = self._predicted_values(X)

        return np.nanmean(predicted_values, axis=1)

    def predict_avevar(self, X):

        predicted_values = self._predicted_values(X)

        return (
            np.nanmean(predicted_values, axis=1),
            np.nanvar(predicted_values, axis=1)
        )