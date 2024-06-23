import pandas as pd
import numpy as np

class ClassifyAB():

    def __init__(self, classification_func=None) -> None:
        
        self.classification_func = self.k_means if classification_func is None else classification_func 

    def reduce_mem_usage(self, df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2    
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df

    def use_cfunc(self, new_func=None, cmethod: str=None):
        if new_func is not None:
            self.classification_func = new_func
        else:
            if cmethod is None:
                raise ValueError("new_func=None时， cmethod不能为空， 可在['kmeans', 'meanshift', 'dbscan']选填")
            if cmethod.lower().repalce('_', '')=='kmeans':   
                self.classification_func = self.k_means
            if cmethod.lower().repalce('_', '')=='meanshift':   
                self.classification_func = self.mean_shift_train        
            if cmethod.lower().repalce('_', '')=='dbscan':   
                self.classification_func = self.DBSCAN_train 

    def mean_shift_train(self, X, bandwidth=0.95, **kargs):
        from sklearn.cluster import MeanShift, estimate_bandwidth
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=8, **kargs)
        ms.fit(X)
        labels = ms.labels_
        return labels

    def DBSCAN_train(self, X, **kargs):
        from sklearn.cluster import DBSCAN
        db = DBSCAN(**kargs).fit(X)
        labels = db.labels_
        return labels


    def k_means(self, data, n_clusters=100, **kargs):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        k_means_cluster = KMeans(n_clusters, random_state=0, **kargs).fit(data)
        
        score = silhouette_score(data, k_means_cluster.labels_)
        print(score)
        
        labels =  k_means_cluster.labels_
        
        return labels
        

    def train_label(self, df, OneHutColumn: list=None, StdScaler=True, **kargs):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        train_data = self.reduce_mem_usage(df)
        if OneHutColumn is not None:
            train_data = pd.get_dummies(train_data, columns=OneHutColumn)
        else:
            OneHutColumn = []
            for column, dtype in zip(df.columns, df.dtypes.map(str)):
                if dtype in ['object', 'string']:
                    OneHutColumn.append(column)
            if len(OneHutColumn)!=0:
                train_data = pd.get_dummies(train_data, columns=OneHutColumn)

        if StdScaler:
            scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            scaler = MinMaxScaler()
        train_data = scaler.fit(train_data).transform(train_data)
        
        label = self.classification_func(train_data,  **kargs)
        return label

    def split_ab_group(self, data, label_column='label', control_columns=None, max_iter=1000, n=5, use_sample=0.7, e=0.01):
        '''
        Parameters
        ----------
        data : DataFrame
            打上聚类标签的数据集
        label_column : str
            聚类得出的标签, 默认为`label`   
        control_columns : List
            需要精确控制误差的特征
        max_iter : Int
            最大抽样次数
        n : Int
            需要抽几组
        use_sample : (0, 1)
            需要使用全部样本的多少，越小误差越越小
        e : (0, 1)
            误差率
        Returns
        -------
        group_list : List[DataFrame]

        '''
        index_list = []
        group_list = []
        control_columns = data.columns if control_columns is None else control_columns
        label_num = data[label_column].value_counts()
        assert label_num.min()*use_sample>n*1.5
        label_num = (label_num*use_sample/n).map(int).to_dict()
        mean = [data.groupby(label_column, as_index=False).apply(lambda df: df.sample(label_num[df.name]))[control_columns].sum() for i in range(100)]
        mean = pd.concat(mean, axis=1).mean(axis=1)
        i = 1
        while len(group_list)<n and i < max_iter:
            temp =  data.loc[~data.index.isin(index_list)].groupby(label_column, as_index=False).apply(lambda df: df.sample(label_num[df.name]))
            ts = temp[control_columns].sum()
            var = (ts - mean).abs() / mean
            if (var < e).all():
                temp.index = temp.index.droplevel(0)
                index_list.extend(temp.index.to_list())
                group_list.append(temp)
            i += 1
        return group_list