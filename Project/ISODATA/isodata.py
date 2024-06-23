import numpy as np

class Isodata():
    def __init__(self, k, split_dist, merge_dist, max_iter, min_sample=1):
        self.k = k
        self.min_sample = min_sample
        self.split_dist = split_dist
        self.merge_dist = merge_dist
        self.max_iter = max_iter
        self.data = None
        self.size = None
        self.dim = None
        self.weight = None
        self.n_centers = None
        self.center_ids = None
        self.labels_ = None


    def fit(self, data, weight):
        self.data = data.values
        self.size = data.shape[0]
        self.dim = data.shape[1]
        self.weight = np.array(weight)
        self.n_centers = self.data[np.random.choice(range(self.size), self.k, replace=False)].copy()
        
        for iter in range(self.max_iter):
            self.divide()
            self.get_new_center()
            if iter%2==1:
                self.split()
            else:
                self.merge()
        self.get_labels()

    def divide(self):
        center_ids = [[] for _ in range(self.n_centers.shape[0])]
        for i in range(self.size):
            dist = np.sum(np.abs(self.data[i]-self.n_centers), axis=1)
            min_id = np.argmin(dist)
            center_ids[min_id].append(i)
        self.n_centers = self.n_centers[[len(ids)>0 for ids in center_ids]]
        self.center_ids = [ids for ids in center_ids if len(ids)>0]

        
    def get_new_center(self):
        n_centers = []
        for ids in self.center_ids:
            # center = np.mean(self.data[ids], axis=0)
            max_vector_id = np.argmax(np.sum(self.data[ids], axis=1))
            center = self.data[ids[max_vector_id]]
            n_centers.append(center)
        self.n_centers = np.array(n_centers)
    
    def get_max_dist(self, ids):
        max_vector_id = np.argmax(np.sum(self.data[ids], axis=1))
        max_vector = self.data[ids[max_vector_id]]
        dist = np.sum(np.abs(max_vector-self.data[ids]), axis=1)
        return np.max(dist)

    def split(self):
        n_centers = []
        for cenrter, ids in zip(self.n_centers, self.center_ids):
            max_dist = self.get_max_dist(ids=ids)
            if max_dist>self.split_dist:
                min_vector_id = np.argmin(np.sum(self.data[ids], axis=1))
                # n_centers.append(cenrter+max_dist/len(ids)/2)
                # n_centers.append(cenrter-max_dist/len(ids)/2)
                n_centers.append(cenrter)
                n_centers.append(self.data[ids[min_vector_id]])
            else:
                n_centers.append(cenrter)
        self.n_centers = np.array(n_centers)

    def merge(self):
        n_centers = []
        used_centers = []
        for i in range(self.n_centers.shape[0]):
            if i in used_centers:
                continue
            new_center = None
            for j in range(i, self.n_centers.shape[0]):
                if j in used_centers or i==j:
                    continue
                dist = np.sum(np.abs(self.n_centers[i]-self.n_centers[j]))
                if dist<=self.merge_dist:
                    new_center = (self.n_centers[i]+self.n_centers[j])/2
                    n_centers.append(new_center)
                    used_centers.append(i)
                    used_centers.append(j)
                    break
            if new_center is None:
                n_centers.append(self.n_centers[i])
                used_centers.append(i)
        self.n_centers = np.array(n_centers)

    def get_labels(self):
        self.divide()
        self.get_new_center()
        label_map = [(i, id) for i in range(len(self.center_ids)) for id in self.center_ids[i]]
        label_map.sort(key=lambda x: x[1])
        self.labels_ = [_[0] for _ in label_map]


if __name__=='__main__':
    import pandas as pd
    df = pd.read_excel('')
    train_data = pd.DataFrame(data=[[int(_) for _ in i] for i in df.ship_model])
    iso = Isodata(k=50, split_dist=2, merge_dist=2, max_iter=50)
    iso.fit(train_data, df.weight.values)