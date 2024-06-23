
# Visualization with Matplotlib

## Fundament 
### Global Settings
#### 中文乱码
```py
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
```
#### 符号乱码
```py
plt.rcParams['axes.unicode_minus'] = False
```
### Charts



#### subplots
``` py
fig, ax = plt.subplots(2, 2, figsize=(15, 15), facecolor='w')
bar1 = ax[0][0].bar(x1, y1)
ax[0][0].bar_label(bar1, fontsize=16)
ax[0][0].set_title('x1_title')

bar2 = ax[0][1].bar(x2, y2)
ax[0][1].bar_label(bar2, fontsize=16)
ax[0][1].set_title('x2_title')

bar3 = ax[1][0].bar(x3, y3)
ax[1][0].bar_label(bar3, fontsize=16)
ax[1][0].set_title('x3_title')

bar4 = ax[1][1].bar(x4, y4)
ax[1][1].bar_label(bar4, fontsize=16)
ax[1][1].set_title('x4_title')

# or use flatten()
```


## Special case


### 线图添加多个系列
``` py
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(series1, label='label1', linewidth=3)
ax.plot(series2, label='label2', linewidth=3)
plt.legend(fontsize=16)
plt.title('title',fontsize=18)
```

### 组合柱状图、文本相对位置
``` py
fig, ax = plt.subplots(1, 2, figsize=(15,8))
ax[0].bar([i-0.35/2 for i in range(6)], sorted([i.quantity.sum() for i in low_group]), width=0.35, label='low_gropup')
ax[0].bar([i+0.35/2 for i in range(6)], sorted([i.quantity.sum() for i in high_group]), width=0.35, label='high_group')
ax[0].set_title('quantity gap', fontsize=18)
ax[0].text(0.1, 0.95, 'mean gap{}'.format(round(np.mean([low_group[i].quantity.sum()/high_group[i].quantity.sum() for i in range(6)]), 2)), transform=ax[0].transAxes, fontsize=16)
ax[0].legend(fontsize=14)

ax[1].bar([i-0.35/2 for i in range(6)], sorted([i.profit.sum() for i in low_group]), width=0.35, label='low_gropup')
ax[1].bar([i+0.35/2 for i in range(6)], sorted([i.profit.sum() for i in high_group]), width=0.35, label='high_group')
ax[1].set_title('profit gap', fontsize=18)
ax[1].text(0.1, 0.95, 'mean gap{}'.format(round(np.mean([low_group[i].profit.sum()/high_group[i].profit.sum() for i in range(6)]), 2)), transform=ax[1].transAxes, fontsize=16)
ax[1].legend(fontsize=14)
```

### 饼图设置显示格式
``` py
plt.rcParams['font.size'] = 16
fig, ax = plt.subplots(1,2, figsize=(15,15), facecolor='w')
temp1 = temp.loc[(pd.notnull(temp.segment))&(temp['split2']=='low'), 'segment'].value_counts()
temp2 = temp.loc[(pd.notnull(temp.segment))&(temp['split2']=='high'), 'segment'].value_counts()
patches,l_text,p_text = ax[0].pie(
    temp1.values,
    labels=temp1.index,
    autopct='%1.2f%%')
for text in p_text:
    text.set_size(16)
patches,l_text,p_text = ax[1].pie(
    temp2.values,
    labels=temp2.index,
    autopct='%1.2f%%')
ax[0].set_title('low')
ax[1].set_title('high')
for text in p_text:
    text.set_size(16)
```
### 在Pandas中使用子图
``` py
fig, ax = plt.subplots(figsize=(10, 6))
df.groupby('week')['qty'].sum().plot(ax=ax, label='sales')
df.groupby('week')['profit'].sum().plot(ax=ax, secondary_y=True, label='profit')
fig.legend(fontsize=12)
```

### 帕累托图
``` py
fig, ax = plt.subplots(figsize=(12, 6))
temp = (data['output_weeks_after']+data['output_weeks_before']).value_counts()
temp.plot(ax=ax)
ax2 = ax.twinx()
ax2 = (temp.cumsum()/temp.sum()*100).plot(marker='D', color="C1", kind='line', ax=ax, secondary_y=True, zorder=1)
ax2.set_ylim([30,110])
ax2.scatter(10, (temp.cumsum()/temp.sum()*100)[10], color='r', s=80, zorder=2)
ax2.text(10, (temp.cumsum()/temp.sum()*100)[10]+2, s='{:.2%}'.format((temp.cumsum()/temp.sum())[10]), size=16)
```

### 在线图中突出显示某一点
``` py
fig, ax = plt.subplots(figsize=(12, 6))
temp = data.groupby('output_weeks')[['mean_profit_after', 'mean_profit_before']].mean()
(temp.mean_profit_after/temp.mean_profit_before).round(4).plot(ax=ax, zorder=1)
out_point = (temp.mean_profit_after/temp.mean_profit_before).loc[lambda x: x>1]
for index, value in out_point.items():
    ax.scatter(index, value, color='r', s=80, zorder=2)
    ax.text(index-2, value-0.1, s='{:.4}'.format(value), size=16)
```

### 多图联合带关联箭头&玫瑰饼图
``` py
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

x = [0.51, 0.29, 0.2]
df_polar = pd.DataFrame(data={'x':x})
df_polar['area'] = df_polar.x*2*np.pi
df_polar['area_sum'] = df_polar.area.cumsum()
df_polar['theta'] = df_polar.area_sum-df_polar.area/2
df_polar['quantity'] = [0.55, 0.41, 0.04]
label = ['label1', 'label2', 'label3']
theta = df_polar.theta
radii = df_polar.quantity
width = df_polar.area
# colors = np.random.random((len(df_polar),3))
colors = np.array([[0.9385116 , 0.01007244, 0.6481479 ],
       [0.75928194, 0.9580445 , 0.04781437],
       [0.13161391, 0.93843477, 0.10462454]])

fig = plt.figure(facecolor='white', figsize=(15, 8))
ax1 = fig.add_subplot(121, projection='polar')
ax1.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5, edgecolor='w')
ax1.spines['polar'].set_visible(False)
tthetas = [theta[0]+0.05, theta[1]-0.2, theta[2]+0.2]
r1 = [0.3, 0.32, 0.4]
r2 = [0.2, 0.2, 0.3]
for i in range(len(theta)):
    ax1.text(tthetas[i], r1[i], 'qty: {:.0%}'.format(listing[i]), ha='center', va='center',fontsize=16)
    ax1.text(tthetas[i], 0.62, label[i], ha='center', va='center',fontsize=22)
    if i==2:
        tthetas[i] += 0.1 
    ax1.text(tthetas[i], r2[i], 'quantity: {:.0%}'.format(df_polar['quantity'].values[i]), ha='center', va='center',fontsize=16)
    
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('title', fontdict={'fontsize':22}, pad=40)

# 绘制第二个子图
ax2 = fig.add_subplot(122)
age_ratios = [.93, .05, .02]
age_labels = ['age_label1', 'age_label2', 'age_label3']
bottom = 1
width = .2
for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
                 alpha=0.1 + 0.25 * (j+1))
    ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

ax2.set_title('title2', fontdict={'fontsize':22}, pad=10)
ax2.legend(loc='upper right')
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 3 * width)

# 添加指向箭头
from matplotlib.patches import ConnectionPatch, ArrowStyle
ns = ArrowStyle("simple", head_length=0.8, head_width=0.8, tail_width=0.3)
ax2.annotate("",
            xy=(-0.15, 0.25), xycoords='data',
            xytext=(-1.1, 0.2), textcoords='data',
            arrowprops=dict(arrowstyle=ns, connectionstyle="arc3", edgecolor='dodgerblue'))


plt.show()
```


### 极坐标带标记

``` py
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
r = np.arange(0, 1, 0.001)
theta = 2 * 2*np.pi * r
line, = ax.plot(theta, r, color='#ee8d18', lw=1)

ind = 800
thisr, thistheta = r[ind], theta[ind]
ax.plot([thistheta], [thisr], 'o')
ax.annotate('a polar annotation',
            xy=(thistheta, thisr),  # theta, radius
            xytext=(0.05, 0.05),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(facecolor='black', shrink=0.01, width=2),
            horizontalalignment='left',
            verticalalignment='bottom')
```

### 设置标签显示数量

``` py
from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(5))
```

### 轴标签旋转
``` py
plt.xticks(rotation=30)
fig.autofmt_xdate(rotation=45)
ax.set_xticklabels(xlabels, rotation=45, ha="right")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.tick_params(axis="x", labelrotation=45)
```

### 隐藏轴标签
``` py
import matplotlib.pyplot as plt

plt.plot([0, 10], [0, 10])
plt.xlabel("X Label")
plt.ylabel("Y Label")

ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.grid(True)
plt.show()
```

### 隐藏轴标题
``` py
fig, ax = plt.subplots(1, 2, figsize=(12, 6), facecolor='w')
sns.boxplot(data=temp2, x='shipping_large', y='final_freight', ax=ax[0])
ax[0].set_xlabel('')
sns.boxplot(data=temp2, x='shipping_large', y='profit', ax=ax[1])
ax[1].set_xlabel('')
```
### 设置坐标轴显示范围
``` py
ax2.set_xlim(low, high)
```