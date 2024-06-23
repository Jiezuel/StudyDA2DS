# Visualization with Seaborn

## Fundament 
### Global Settings

### Charts
#### 分类计数柱状图
``` py
sns.countplot(x='class', data=df)
```

#### 双变量密度分布图
``` py
sns.jointplot(x=variable1, y=variable2, kind='kde') # or kind='regg'
```
## Special case