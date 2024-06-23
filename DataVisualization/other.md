## xx


## 词云图
``` py 
from wordcloud import WordCloud

wcd=WordCloud(font_path='./xxx/bb.tff')

text='富强 民主 文明 5 脉动 max 脉动 脉动 脉动 脉动'

wcd.repeat=True
wcd.generate(text)

wcd.to_image()

```