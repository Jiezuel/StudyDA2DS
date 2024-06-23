import pandas as pd

class Item:
    def __init__(self, data, products):
        self.data = data
        self.products = set(products)
    
class Product:
    def __init__(self, data, items):
        self.data = data
        self.items = items

class ItemProductTree:
    def __init__(self, threshold=0.8):
        self._root = None
        self.depth = 0
        self.items = []
        self.products = set()
        self.threshold = threshold
        self.used_products = set()

    def similarity(self, sku):
        same_num = len(sku.products & self.products)
        a = same_num/len(sku.products)
        b = same_num/len(self.products)
        s = max(a, b)
        return s>=self.threshold

    def update(self, sku):
        self.items.append(sku.data)
        self.products.update(sku.products)

    def get_alive_product(self, parents):
        if len(parents)>0:
            self.used_products.update(parents)
            return list(self.products-self.used_products)
        else:
            return list(self.products)

    def add(self, node):
        items = []
        if self._root is None:
            assert isinstance(node, Item)
            self._root = node
            self.update(node)
        else:
            assert isinstance(node, list)
            for sku in node:
                if sku.data in self.items:
                    continue

                if self.similarity(sku):
                    self.update(sku)
                    items.append(sku.data)
        self.depth+=1
        return items


class ItemProductCluster:

    def __init__(self, threshold=0.8, max_depth=5):

        self.threshold = threshold
        self.max_depth = max_depth
        self.data = None
        self.sku_map = None
        self.product_map = None
        self.clusters = []
        self.labels_ = None


    def fit(self, listing: pd.Series, product_id: pd.Series):
        self.concat(a=listing, b=product_id)
        self.sku2product()
        self.product2sku()
    
    def concat(self, a, b):
        df = pd.concat([a, b], axis=1)
        df.columns = ['sku', 'product_id']
        df.sku = df.sku.astype(str)
        df.product_id = df.product_id.astype(int)
        self.data = df 

    def sku2product(self):
        sku2product_dict = self.data.groupby('sku').apply(lambda x: x.product_id.drop_duplicates().to_list()).to_dict()
        for key, value in sku2product_dict.items():
            sku2product_dict[key] = Item(key, value)
        self.sku_map = sku2product_dict
    
    def product2sku(self):
        product2sku_dict = self.data.groupby('product_id').apply(lambda x: x.sku.drop_duplicates().to_list()).to_dict()
        for key, value in product2sku_dict.items():
            items = [self.sku_map.get(item) for item in value]
            product2sku_dict[key] = Product(key, items)
        self.product_map = product2sku_dict

    def train(self):

        while self.sku_map: # 直达sku_map为空才结束循环，代表着每一个sku_id都被打上了标签

            x = self.sku_map.popitem()[1]  # 取一个Item对象，该对象的值是sku_id, products属性代表了该sku_id包含的产品ID
            tree = ItemProductTree(threshold=self.threshold)  # 初始化一个item-->product搜索树
            tree.add(x) # 向树中添加一个item节点
            parents = [] # 存放一个添加的节点，避免重复添加
            while tree.depth<self.max_depth: # 如果树的深度达到预设值，则结束迭代
                
                alive_products = tree.get_alive_product(parents) # 获取还没有分支的product_id
                if len(alive_products)==0: # 如果所有的product_id都已经搜索过，说明树已经到底，故结束循环
                    break
                parents = alive_products # 接下来对父节点对应的每一个product_id，搜索其对应的sku_id，故先将product_id变为父节点
                for product in alive_products:
                    items = self.product_map.get(product).items # 找到某个product_id对应的sku_id
                    items = [_ for _ in items if self.sku_map.get(_.data)] # 如果sku_id没有被匹配过才进入后一步，这也产生先来后到的问题，如果一个sku_id之前已经被匹配过了，将无法再次匹配
                    cluster_items = tree.add(items) # 一次添加所有未匹配节点，返回符合相似要求的节点
                    [self.sku_map.pop(_) for _ in cluster_items if self.sku_map.get(_)] # 从候选集中剔除已匹配节点

            self.clusters.append(tree) # 一棵树就是一个类
            
        self.get_label() # 迭代clusters，给每一个类一个标签，并标注到对应的sku_id上


    def get_label(self):
        cdata = [tree.items for tree in self.clusters]
        item_label = pd.Series(data=cdata).explode().reset_index()
        item_label.columns = ['label', 'az_item_number']
        clabel_map = item_label.set_index('az_item_number').label.to_dict()
        self.labels_ = self.data.sku.map(clabel_map).values


if __name__=='__main__':

    model = ItemProductCluster()