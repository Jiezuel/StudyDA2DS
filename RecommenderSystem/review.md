## 经典算法
### 协同过滤
#### 协同过滤基于用户
#### 协同过滤基于物品
 $$ sim_{uv}=\frac{|N(u) \cap N(v)|}{|N(u)| \cup|N(v)|} $$

 $$ sim_{uv} = cos(u,v) =\frac{u\cdot v}{|u|\cdot |v|} $$


 $$ sim_{uv} = \frac{\sum_i r_{ui}*r_{vi}}{\sqrt{\sum_i r_{ui}^2}\sqrt{\sum_i r_{vi}^2}} $$

 $$        r = \frac{\sum (x - m_x) (y - m_y)}
                 {\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}} \\
                 
                 f(r) = \frac{{(1-r^2)}^{n/2-2}}{\mathrm{B}(\frac{1}{2},\frac{n}{2}-1)}$$

 $$ R_{\mathrm{u}, \mathrm{p}}=\frac{\sum_{\mathrm{s} \in S}\left(w_{\mathrm{u}, \mathrm{s}} \cdot R_{\mathrm{s}, \mathrm{p}}\right)}{\sum_{\mathrm{s} \in S} w_{\mathrm{u}, \mathrm{s}}} $$    

 $$ R_{\mathrm{u}, \mathrm{p}}=\bar{R}{u} + \frac{\sum{\mathrm{s} \in S}\left(w_{\mathrm{u}, \mathrm{s}} \cdot \left(R_{s, p}-\bar{R}{s}\right)\right)}{\sum{\mathrm{s} \in S} w_{\mathrm{u}, \mathrm{s}}} $$

 $$ \operatorname{sim}(\text { Alice, user1 })=\cos (\text { Alice, user } 1)=\frac{15+3+8+12}{\operatorname{sqrt}(25+9+16+16) * \operatorname{sqrt}(9+1+4+9)}=0.975 $$

$$ \operatorname{Recall}=\frac{\sum_{u}|R(u) \cap T(u)|}{\sum_{u}|T(u)|} $$


 $$ \text { Coverage }=\frac{\left|\bigcup_{u \in U} R(u)\right|}{|I|} $$
## 深度学习算法

## 现实复杂场景算法