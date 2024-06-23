from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


class RFRegressor(object):
    """Random Forest Regressor
    """
    
    
    def __init__(self, n_trees, sample_sz, min_leaf_sz=5, n_jobs=None, max_depth=None):
        self._n_trees = n_trees
        self._sample_sz = sample_sz
        self._min_leaf_sz = min_leaf_sz
        self._n_jobs = n_jobs
        self._max_depth = max_depth
        self._trees = [self._create_tree() for i in range(self._n_trees)]
        
    def _get_sample_data(self, bootstrap=True):
        """Generate training data for each underlying decision tree
        
        Parameters
        ----------
        bootstrap: boolean value, True/False
            The default value is True, it would bootstrap sample from 
            input training data. If False, the exclusive sampling will
            be performed.
            
        Returns
        -------
        idxs: array-like object
            Return the indices of sampled data from input training data
        """
        if bootstrap:
            idxs = np.random.choice(len(self._X), self._sample_sz)
        else:
            idxs = np.random.permutation(len(self._X))[:self._sample_sz]
        return idxs
    
    def _create_tree(self):
        """Build decision treee
        
        Returns
        -------
        dtree : DTreeRegressor object
        """
        return DTreeRegressor(self._min_leaf_sz, self._max_depth)
    
    def _single_tree_fit(self, tree):
        """Fit the single underlying decision tree
        
        Parameters
        ----------
        tree : DTreeRegressor object
        
        Returns
        -------
        tree : DTreeRegressor object
        """
        sample_idxs = self._get_sample_data()
        return tree.fit(self._X.iloc[sample_idxs], self._y[sample_idxs])
        
    def fit(self, x, y):
        """Train a forest regressor of trees from the training set(x, y)
        
        Parameters
        ----------
        x : DataFrame,
            The training input samples.
            
        y : Series or array-like object
            The target values.
        
        """
        self._X = x
        self._y = y
        if self._n_jobs:
            self._trees = self._parallel(self._trees, self._single_tree_fit, self._n_jobs)
        else:
            for tree in self._trees:
                self._single_tree_fit(tree)
    
    def predict(self, x):
        """Predict target values using trained model
        
        Parameters
        ---------
        x : DataFrame or array-like object
           input samples
           
        Returns
        -------
        ypreds : array-like object
            predicted target values
        """
        all_tree_preds = np.stack([tree.predict(x) for tree in self._trees]) 
        return np.mean(all_tree_preds, axis=0)
    
    def _parallel(self, trees, fn, n_jobs=1):
        """Parallel jobs execution
        
        Parameters
        ----------
        trees : list-like object
            a list-like object contains all underlying trees
            
        fn : function-like object
            map function
        
        n_jobs : integer
            The number of jobs.
            
        Returns
        -------
        result : list-like object
            a list-like result object for each call of map function `fn`
        """
        try:
            workers = cpu_count()
        except NotImplementedError:
            workers = 1
        if n_jobs:
            workders = n_jobs
        pool = Pool(processes=workers)
        result = pool.map(fn, trees)
        return list(result)
    
    @property
    def feature_importances_(self):
        """Calculate feature importance
        
        Returns
        -------
        self._feature_importances : array-like object
            the importance score of each feature 
        """
        if not hasattr(self, '_feature_importances'):
            norm_imp = np.zeros(len(self._X.columns))
            for tree in self._trees:
                t_imp = tree.calc_feature_importance()
                norm_imp = norm_imp + t_imp / np.sum(t_imp)
            self._feature_importances = norm_imp / self._n_trees
        return self._feature_importances
       
    @property
    def feature_importances_extra(self):
        """Another method to calculate feature importance
        """
        norm_imp = np.zeros(len(self._X.columns))
        for tree in self._trees:
            t_imp = tree.calc_feature_importance_extra()
            norm_imp = norm_imp + t_imp / np.sum(t_imp)
        norm_imp = norm_imp / self._n_trees
        imp = pd.DataFrame({'col':self._X.columns, 'imp':norm_imp}).sort_values('imp', ascending=False)
        return imp
    
    
class DTreeRegressor(object):
    
    
    def __init__(self, min_leaf_sz, max_depth=None):
        self._min_leaf_sz = min_leaf_sz
        self._split_point = 0
        self._split_col_idx = 0
        self._score = float('inf')
        self._sample_sz = 0
        self._left_child_tree = None
        self._right_child_tree = None
        self._feature_importances = []
        self._node_importance= 0
        if max_depth is not None:
            max_depth = max_depth - 1
        self._max_depth = max_depth
        
    def fit(self, x, y):
        self._X = x
        self._y = y
        self._col_names = self._X.columns
        self._feature_importances = np.zeros(len(self._col_names))
        self._sample_sz = len(self._X)
        self._val = np.mean(self._y)
        if self._max_depth is not None and self._max_depth < 2:
            return self
        self._find_best_split()
        return self
        
    def _calc_mse_inpurity(self, y_squared_sum, y_sum, n_y):
        """Calculate Mean Squared Error impurity
        
        This is just the recursive version for calculating variance
        
        Parameters
        ----------
        y_squared_sum: float or int , the sum  of y squared 
        
        y_sum: float or int , the sum of y value
        
        n_y: int, the number of samples
        
        
        Returns
        -------
        
        """
        dev = (y_squared_sum / n_y) - (y_sum / n_y) ** 2
        return dev
    
    def _find_best_split(self):
        for col_idx in range(len(self._col_names)):
            self._find_col_best_split_point(col_idx)
            
        self._feature_importances[self._split_col_idx] = self._node_importance
        
        if self.is_leaf:
            return 
        
        left_child_sample_idxs = np.nonzero(self.split_col <= self.split_point)[0]
        right_child_sample_idxs = np.nonzero(self.split_col > self.split_point)[0]
        
        self._left_child_tree = (DTreeRegressor(self._min_leaf_sz, self._max_depth)
                                 .fit(self._X.iloc[left_child_sample_idxs], self._y[left_child_sample_idxs]))
        self._right_child_tree = (DTreeRegressor(self._min_leaf_sz, self._max_depth)
                                  .fit(self._X.iloc[right_child_sample_idxs], self._y[right_child_sample_idxs]))
            
    def _find_col_best_split_point(self, col_idx):
        x_col = self._X.values[:, col_idx]
        sorted_idxs = np.argsort(x_col)
        sorted_x_col = x_col[sorted_idxs]
        sorted_y =  self._y[sorted_idxs]
        
        lchild_n_samples = 0
        lchild_y_sum  = 0.0
        lchild_y_square_sum = 0.0

        rchild_n_samples = self._sample_sz
        rchild_y_sum = sorted_y.sum()
        rchild_y_square_sum = (sorted_y ** 2).sum()
        
        node_y_sum = rchild_y_sum
        node_y_square_sum = rchild_y_square_sum 
        for i in range(0, self._sample_sz - self._min_leaf_sz):
            xi, yi = sorted_x_col[i], sorted_y[i]
            
            rchild_n_samples -= 1
            rchild_y_sum -= yi
            rchild_y_square_sum -= (yi ** 2)
            
            lchild_n_samples  +=  1
            lchild_y_sum += yi
            lchild_y_square_sum += (yi ** 2)
            
            if i < self._min_leaf_sz  or xi == sorted_x_col[i+1]:
                continue
            
            lchild_impurity = self._calc_mse_inpurity(lchild_y_square_sum,
                                                      lchild_y_sum, lchild_n_samples)
            rchild_impurity = self._calc_mse_inpurity(rchild_y_square_sum,
                                                      rchild_y_sum, rchild_n_samples)
            split_score  = (lchild_n_samples * lchild_impurity 
                            + rchild_n_samples * rchild_impurity) / self._sample_sz
            
            if split_score < self._score:
                self._score = split_score
                self._split_point = xi
                self._split_col_idx = col_idx
                self._node_importance = (self._sample_sz 
                    * (self._calc_mse_inpurity(node_y_square_sum, node_y_sum, self._sample_sz) 
                    - split_score))
                    
    
    def predict(self, x):
        if type(x) == pd.DataFrame:
            x = x.values
        return np.array([self._predict_row(row) for row in x])
    
    def _predict_row(self, row):
        if self.is_leaf:
            return self._val
        t = (self._left_child_tree if row[self._split_col_idx] 
             <= self.split_point else self._right_child_tree)
        return t._predict_row(row)

    def __repr__(self):
        pr =  f'sample: {self._sample_sz}, value: {self._val}\r\n'
        if not self.is_leaf:
            pr += f'split column: {self.split_name}, \
                split point: {self.split_point}, score: {self._score} '
        return pr
    
    def calc_feature_importance(self):
        if self.is_leaf:
            return self._feature_importances
        return (self._feature_importances 
                + self._left_child_tree.calc_feature_importance()
                + self._right_child_tree.calc_feature_importance()
               )
    
    def calc_feature_importance_extra(self):
        imp = []
        o_preds = self.predict(self._X.values)
        o_r2 = metrics.r2_score(self._y, o_preds)
        for col in self._col_names:
            tmp_x = self._X.copy()
            shuffle_col = tmp_x[col].values
            np.random.shuffle(shuffle_col)
            tmp_x.loc[:,col] = shuffle_col
            tmp_preds = self.predict(tmp_x.values)
            tmp_r2 = metrics.r2_score(self._y, tmp_preds)
            imp.append((o_r2 - tmp_r2))
        imp = imp / np.sum(imp)
        return imp
    
    @property
    def split_name(self):
        return self._col_names[self._split_col_idx]
    
    @property
    def split_col(self):
        return self._X.iloc[:, self._split_col_idx]

    @property
    def is_leaf(self):
        return self._score == float('inf')
    
    @property
    def split_point(self):
        return self._split_point