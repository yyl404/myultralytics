# 面向跨任务双向分布外泛化失效的类增量目标检测方法

## 1. 版本说明

本文档整理当前讨论形成的基础版本方法，统一采用以下设计：

1. **原型可靠性感知的 YOLO-LwF 蒸馏**：取代原先的单通道 Confidence-Aware Distillation（CAD），分别处理教师前景、背景和回归输出。
2. **基于局部 patch 原型分布采样的全类别回放蒸馏**：重建历史旧类前景 patch，通过教师检测头获得旧类分类与回归响应，并为新类别补充背景目标，对学生全类别分类输出和回归输出进行蒸馏。
3. **特征值缩放投影正则化 ESPReg**：稳定骨干和颈部网络，降低检测头输入特征及历史原型的漂移。
4. **固定共享 PCA**：基础阶段为每个检测尺度建立一组共享 PCA，后续阶段固定使用；本版本不实现扩张式 PCA，也不进行阶段性 PCA 重编码。
5. **局部多中心分布**：类别内部采用自动确定中心数量的多中心原型，并为每个局部成分估计协方差。基础实现可优先采用对角协方差。

此前的独立 HBC 损失不再作为单独模块；其新类背景校准作用已经并入全类别回放蒸馏。

## 2. 研究动机

类增量目标检测通常利用知识蒸馏保留历史类别知识，并以经验回放作为补充。本文关注任务阶段间图像域分布存在差异时出现的两类方向相反的分布外泛化失效：

- **基础模型旧类知识向增量任务分布泛化失效**：基础模型作为教师时，可能在当前训练图像上将新类目标或新域背景高置信地误判为旧类，形成不可靠蒸馏监督。
- **增量模型新类知识向基础任务分布泛化失效**：新类决策边界向旧域不合理外推，使增量模型在历史测试图像上将旧类目标高置信地误报为新类。

因此，跨任务混淆不仅表现为旧知识遗忘，还表现为新旧类别知识在彼此图像域上的错误泛化。仅依赖教师自身置信度筛选输出，无法充分识别第一类失效；仅对当前图像进行旧类蒸馏，也不能直接约束第二类失效。

三个方法模块的作用分别为：

$$
\text{原型匹配与前景／背景分离}
\longrightarrow
\text{提高真实增量图像上的教师监督可靠性},
$$

$$
\text{历史局部前景 patch 的全类别回放蒸馏}
\longrightarrow
\text{保持旧类响应并校准新类边界},
$$

$$
\text{ESPReg}
\longrightarrow
\text{稳定历史原型所在的检测头输入特征空间}.
$$

## 3. 问题定义与符号

在第 $t$ 个增量阶段，模型只能访问当前训练集 $\mathcal D_t$ 和当前类别集合 $\mathcal V_t$。定义：

$$
\mathcal V_{\mathrm{old}}
=
\mathcal V_1\cup\cdots\cup\mathcal V_{t-1},
\qquad
\mathcal V_{\mathrm{new}}=\mathcal V_t,
$$

$$
\mathcal V_{\mathrm{all}}
=
\mathcal V_{\mathrm{old}}\cup\mathcal V_{\mathrm{new}}.
$$

基础模型 $m_{t-1}$ 已学习旧类别。增量模型 $m_t$ 由 $m_{t-1}$ 初始化，并扩展分类输出通道，使其能够检测 $\mathcal V_{\mathrm{all}}$。

训练过程中，$m_{t-1}$ 作为冻结教师，$m_t$ 作为学生。设检测器共有 $S$ 个检测尺度，第 $s$ 个尺度的检测头输入特征为：

$$
X_s\in\mathbb R^{C_s\times H_s\times W_s}.
$$

本文所有类别原型均构建在**检测头输入处**，即颈部网络输出、分类分支和回归分支共同接收的特征，而不是最终分类层之前的单向量特征。

## 4. 检测头输入局部前景 patch 的提取

### 4.1 为什么使用 patch 而不是单向量

检测头内部包含 $3\times3$ 等空间卷积，因此某个位置的中心输出同时依赖多个邻近位置。若原型位于检测头输入处，单个坐标向量不足以重现该位置的分类和回归计算，必须保存覆盖检测头局部感受野的 patch。

### 4.2 patch 的空间范围

对于正样本位置 $k=(i,j)$，提取：

$$
\boxed{
P_{s,k}
=
X_s\left[:,\ i-r_s:i+r_s+1,\ j-r_s:j+r_s+1\right]
}
$$

即：

$$
P_{s,k}\in
\mathbb R^{C_s\times(2r_s+1)\times(2r_s+1)}.
$$

若检测头第 $l$ 个空间卷积层的卷积核大小为 $k_l$、膨胀率为 $d_l$、步长为 1，则中心输出所需的近似半径为：

$$
r_s=\sum_l\frac{d_l(k_l-1)}{2}.
$$

若分类分支和回归分支所需半径不同，则取：

$$
r_s=\max\left(r_s^{\mathrm{cls}},r_s^{\mathrm{reg}}\right).
$$

例如，连续两层无膨胀的 $3\times3$ 卷积通常需要 $r_s=2$，即提取 $5\times5$ patch。

### 4.3 每个 GT 实例的代表性位置

一个 GT 实例通常对应多个相邻正样本位置，其 patch 高度重叠。为避免同一实例被重复计数，每个实例在每个有效尺度上选择一个代表性位置。

设 GT 实例 $b$ 在尺度 $s$ 上的正样本位置集合为 $\Omega_s(b)$，选择：

$$
k_{b,s}=\arg\max_{k\in\Omega_s(b)}\alpha_{b,k},
$$

其中 $\alpha_{b,k}$ 是标签分配器给出的对齐分数。可进一步将候选位置限制在 GT 中心区域，减少目标边缘和背景的影响。

对应实例级局部前景特征为：

$$
P_{b,s}=\operatorname{Crop}\left(X_s,k_{b,s},r_s\right).
$$

对于靠近特征图边界、无法取得完整 patch 的实例—尺度组合，基础版本直接跳过，避免 padding 区域进入原型分布。

## 5. 基础阶段建立固定共享 PCA

### 5.1 patch 向量化

将实例级 patch 展开为：

$$
p_{b,s}=\operatorname{vec}(P_{b,s})\in\mathbb R^{D_s},
$$

其中：

$$
D_s=C_s(2r_s+1)^2.
$$

由于 $D_s$ 通常较大，直接在原始 patch 空间聚类并估计完整协方差成本较高，因此在每个尺度上建立共享 PCA。

### 5.2 每个尺度建立一组共享 PCA

基础阶段收集尺度 $s$ 上全部基础类别的实例级前景 patch，计算均值：

$$
\bar p_s=\frac1{N_s}\sum_{b=1}^{N_s}p_{b,s}.
$$

对中心化 patch 进行 PCA，保留前 $d_s$ 个主方向，得到：

$$
B_s\in\mathbb R^{D_s\times d_s},
\qquad d_s\ll D_s.
$$

任意 patch 的低维表示为：

$$
\boxed{
h_{b,s}=B_s^\top\left(p_{b,s}-\bar p_s\right)
\in\mathbb R^{d_s}
}
$$

对应重建为：

$$
\widehat p_{b,s}=\bar p_s+B_sh_{b,s}.
$$

基础版本使用非白化投影，因此投影和回放不必单独保存 PCA 特征值。每个尺度保存：

$$
\boxed{\bar p_s,\quad B_s.}
$$

### 5.3 多阶段固定使用

后续阶段新增类别的 patch 使用同一组固定参数：

$$
h_{b,s}=B_s^\top(p_{b,s}-\bar p_s).
$$

因此：

- 不根据新类别重新计算 PCA；
- 不保存阶段性 PCA 版本；
- 不对旧原型进行坐标转换；
- 整个训练过程始终只保存 $S$ 组共享 PCA。

该设计优先保证实现简洁和坐标统一，但不保证基础阶段 PCA 能完整覆盖所有未来类别的重要方向。这是基础版本需要明确承认的近似。

## 6. 多中心类别原型与局部分布

### 6.1 自动确定中心数量

对于类别 $c$ 在尺度 $s$ 上的低维实例特征集合：

$$
\mathcal H_c^{(s)}
=
\{h_{b,s}\mid\operatorname{class}(b)=c\},
$$

采用带最小簇约束的 DP-Means 自动确定中心数量。其目标为：

$$
\min\sum_i\left\|h_i-\mu_{a_i}\right\|_2^2+\lambda_c M,
$$

其中 $M$ 为实际形成的中心数量，$\lambda_c$ 为新增中心的代价。

对样本 $h_i$，按照以下规则分配：

$$
a_i=
\begin{cases}
\arg\min_m\|h_i-\mu_m\|_2,
& \min_m\|h_i-\mu_m\|_2\le\tau_{\mathrm{new}},\\[4pt]
M+1,
& \min_m\|h_i-\mu_m\|_2>\tau_{\mathrm{new}}.
\end{cases}
$$

第二种情况表示当前样本与已有中心均不够接近，因此创建新中心。实际实现还应：

- 删除样本数低于 $n_{\min}$ 的噪声簇；
- 合并距离过近的中心；
- 必要时根据存储预算限制中心数量。

### 6.2 局部原型参数

保留下来的第 $m$ 个局部成分记为：

$$
\left(\mu_{c,m}^{(s)},C_{c,m}^{(s)},\pi_{c,m}^{(s)}\right),
$$

其中：

$$
\mu_{c,m}^{(s)}
=
\frac1{n_{c,m}^{(s)}}
\sum_{h_i\in\mathcal H_{c,m}^{(s)}}h_i,
$$

$$
C_{c,m}^{(s)}
=
\frac1{n_{c,m}^{(s)}}
\sum_{h_i\in\mathcal H_{c,m}^{(s)}}
(h_i-\mu_{c,m}^{(s)})(h_i-\mu_{c,m}^{(s)})^\top,
$$

$$
\pi_{c,m}^{(s)}
=
\frac{n_{c,m}^{(s)}}{\sum_{m'}n_{c,m'}^{(s)}}.
$$

基础实现可优先采用对角协方差：

$$
C_{c,m}^{(s)}
=
\operatorname{diag}\left(
\sigma_{c,m,1}^2,\ldots,\sigma_{c,m,d_s}^2
\right),
$$

也可采用低秩或收缩协方差提高小样本条件下的稳定性。

第 $t-1$ 阶段结束时的历史原型库为：

$$
\mathcal P_{t-1}
=
\left\{
\mu_{c,m}^{(s)},C_{c,m}^{(s)},\pi_{c,m}^{(s)}
\ \middle|\
c\in\mathcal V_{\mathrm{old}}
\right\}.
$$

## 7. 真实增量图像上的原型可靠性感知 YOLO-LwF 蒸馏

该模块不再使用原 CAD 的“最大旧类通道＋单一二值 KL”，而是将教师输出分为可靠旧类前景、可靠旧类背景和不确定区域。

核心原则是：

> **原型匹配只用于前景可靠性评估；背景响应不使用前景原型匹配，也不参与回归蒸馏。**

### 7.1 教师旧类响应

在当前训练图像上，教师和学生的旧类分类概率分别为：

$$
p^T_{k,c}=\sigma(z^T_{k,c}),
\qquad
p^S_{k,c}=\sigma(z^S_{k,c}),
\qquad c\in\mathcal V_{\mathrm{old}}.
$$

教师最大旧类置信度及对应类别为：

$$
q_k=\max_{c\in\mathcal V_{\mathrm{old}}}p^T_{k,c},
$$

$$
c_k^*=\arg\max_{c\in\mathcal V_{\mathrm{old}}}p^T_{k,c}.
$$

### 7.2 教师局部 patch 的原型匹配

提取教师预测位置 $k$ 的检测头输入 patch，并投影到固定 PCA 空间：

$$
h_k^T
=
B_s^\top\left(
\operatorname{vec}(P_{s,k}^T)-\bar p_s
\right).
$$

与教师预测类别的所有局部原型计算最大余弦相似度：

$$
r_k
=
\max_m\cos\left(h_k^T,\mu_{c_k^*,m}^{(s)}\right).
$$

将相似度映射为原型可靠性：

$$
g_k
=
\operatorname{sigmoid}
\left(
\frac{r_k-\tau_{\mathrm{proto}}}{T_{\mathrm{proto}}}
\right).
$$

### 7.3 前景、背景和不确定区域

令 $a_k^{\mathrm{new}}=1$ 表示位置 $k$ 已被当前新类 GT 明确分配。定义：

$$
M_k^F
=
\mathbf 1\left[q_k\ge\tau_F,\ a_k^{\mathrm{new}}=0\right],
$$

$$
M_k^B
=
\mathbf 1\left[q_k\le\tau_B,\ a_k^{\mathrm{new}}=0\right],
\qquad \tau_B<\tau_F.
$$

前景和背景权重分别为：

$$
\boxed{w_k^F=M_k^Fq_k^\gamma g_k}
$$

和：

$$
\boxed{w_k^B=M_k^B(1-q_k)^\gamma.}
$$

由此得到：

- 高教师置信度但原型不匹配的前景响应会被降权；
- 背景响应不需要接近旧类前景原型；
- 当前新类 GT 位置不接受冲突的旧类蒸馏；
- $\tau_B<q_k<\tau_F$ 的不确定区域不进行对应蒸馏；
- 高置信但原型不匹配的位置不会被直接反转为背景监督。

### 7.4 前景旧类分类蒸馏

对可靠前景，根据师生预测框一致性进一步校准分类蒸馏：

$$
A_k
=
\operatorname{stopgrad}
\left[
\operatorname{IoU}\left(\widehat b_k^T,\widehat b_k^S\right)
\right].
$$

定义二值 Bernoulli KL：

$$
D_{\mathrm{BKL}}(p\|q)
=
p\log\frac pq+(1-p)\log\frac{1-p}{1-q}.
$$

前景分类损失为：

$$
\boxed{
\mathcal L_{\mathrm{cls}}^F
=
\frac{
\sum_k w_k^FA_k
\sum_{c\in\mathcal V_{\mathrm{old}}}
D_{\mathrm{BKL}}\left(p^T_{k,c}\middle\|p^S_{k,c}\right)
}{
|\mathcal V_{\mathrm{old}}|\sum_k w_k^FA_k+\varepsilon
}.
}
$$

### 7.5 背景旧类分类蒸馏

背景位置不使用原型权重，也不使用预测框 IoU：

$$
\boxed{
\mathcal L_{\mathrm{cls}}^B
=
\frac{
\sum_k w_k^B
\sum_{c\in\mathcal V_{\mathrm{old}}}
D_{\mathrm{BKL}}\left(p^T_{k,c}\middle\|p^S_{k,c}\right)
}{
|\mathcal V_{\mathrm{old}}|\sum_k w_k^B+\varepsilon
}.
}
$$

完整分类蒸馏为：

$$
\mathcal L_{\mathrm{cls}}^{\mathrm{KD}}
=
\mathcal L_{\mathrm{cls}}^F
+
\eta_B\mathcal L_{\mathrm{cls}}^B.
$$

前景与背景分别归一化，避免数量庞大的背景位置主导损失。必要时可对背景位置进行限量采样。

### 7.6 可靠前景的直接回归蒸馏

对于当前采用的直接边界框回归检测头，令教师和学生的四维回归输出分别为：

$$
\widehat{\mathbf d}_k^T,\quad
\widehat{\mathbf d}_k^S\in\mathbb R^4.
$$

回归蒸馏为：

$$
\boxed{
\mathcal L_{\mathrm{reg}}^{\mathrm{KD}}
=
\frac{
\sum_k w_k^F
\operatorname{SmoothL1}
\left(\widehat{\mathbf d}_k^S,\widehat{\mathbf d}_k^T\right)
}{
\sum_k w_k^F+\varepsilon
}.
}
$$

回归蒸馏只作用于可靠旧类前景，不作用于背景或不确定区域。

真实图像上的完整蒸馏损失为：

$$
\boxed{
\mathcal L_{\mathrm{KD}}
=
\mathcal L_{\mathrm{cls}}^F
+
\eta_B\mathcal L_{\mathrm{cls}}^B
+
\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}^{\mathrm{KD}}.
}
$$

上述公式默认教师和学生预测位置能够直接对齐。对于一对一稀疏预测分支，需要先根据类别和框关系进行匹配，不能直接按预测索引蒸馏；基础实现可先在可对齐的一对多稠密分支上完成。

## 8. 基于原型分布采样的全类别回放蒸馏

该模块不是简单地将历史原型作为新类负样本，而是：

> **将重建 patch 通过教师检测头，获得旧类分类与回归响应；随后为新类别补充背景目标，形成完整的全类别教师监督，对学生的旧类分类、新类分类和回归输出进行联合蒸馏。**

### 8.1 低维局部特征采样

对于旧类别 $c$、尺度 $s$，先选择局部成分：

$$
m\sim
\operatorname{Categorical}
\left(\pi_{c,1}^{(s)},\ldots,\pi_{c,M_c^{(s)}}^{(s)}\right).
$$

若采用对角协方差，采样为：

$$
\boxed{
\widetilde h
=
\mu_{c,m}^{(s)}
+
\sigma_{c,m}^{(s)}\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
}
$$

若采用一般低秩协方差，则：

$$
\widetilde h
=
\mu_{c,m}^{(s)}+L_{c,m}^{(s)}\epsilon,
$$

其中：

$$
C_{c,m}^{(s)}=L_{c,m}^{(s)}L_{c,m}^{(s)\top}.
$$

### 8.2 重建检测头输入 patch

通过固定 PCA 逆变换：

$$
\boxed{\widetilde p=\bar p_s+B_s\widetilde h.}
$$

再恢复空间形状：

$$
\widetilde P
=
\operatorname{reshape}
\left(
\widetilde p,
C_s,2r_s+1,2r_s+1
\right).
$$

### 8.3 教师与学生的回放输出

将重建 patch 分别通过教师和学生检测头，读取中心位置输出。

教师旧类分类与回归输出为：

$$
\mathbf z_{\mathrm{old}}^T
=
H_s^{T,\mathrm{cls}}(\widetilde P)
\in\mathbb R^{|\mathcal V_{\mathrm{old}}|},
$$

$$
\widehat{\mathbf d}^T
=
H_s^{T,\mathrm{reg}}(\widetilde P)
\in\mathbb R^4.
$$

学生全类别分类与回归输出为：

$$
\mathbf z_{\mathrm{all}}^S
=
H_s^{S,\mathrm{cls}}(\widetilde P)
\in\mathbb R^{|\mathcal V_{\mathrm{all}}|},
$$

$$
\widehat{\mathbf d}^S
=
H_s^{S,\mathrm{reg}}(\widetilde P)
\in\mathbb R^4.
$$

### 8.4 构造全类别教师分类目标

教师检测头没有有效的新类输出通道，因此在概率域构造完整教师目标：

$$
\widetilde p_c^T
=
\begin{cases}
\sigma(z_{\mathrm{old},c}^T),
& c\in\mathcal V_{\mathrm{old}},\\[4pt]
0,
& c\in\mathcal V_{\mathrm{new}}.
\end{cases}
$$

学生的全类别分类概率为：

$$
\widetilde p_c^S
=
\sigma(z_{\mathrm{all},c}^S),
\qquad c\in\mathcal V_{\mathrm{all}}.
$$

若在 logit 域表示，则教师目标为：

$$
\widetilde z_c^T
=
\begin{cases}
z_{\mathrm{old},c}^T,
& c\in\mathcal V_{\mathrm{old}},\\[4pt]
-M,
& c\in\mathcal V_{\mathrm{new}},
\end{cases}
$$

其中 $M>0$，使 $\sigma(-M)\approx0$。

**新类背景目标应指概率为 0，或将 logit 设为足够大的负值；不能将原始 logit 设为 0，因为 $\sigma(0)=0.5$。**

### 8.5 PCA 可行性验证与无压缩回退

本方案不再根据教师在重建 patch 上的置信度筛选或加权回放样本。只要采用 PCA 压缩与重建，就默认生成的重建 patch 均可参与回放蒸馏，不再引入额外的回放有效性权重。

这一默认成立的前提是：PCA 重建没有明显改变局部特征经过检测头后的输出。因此，实验中必须单独验证 PCA 近似是否可行。对于同一原始局部 patch $P$，记其展开向量为 $p$，PCA 重建结果为 $\widehat p$，重建 patch 为 $\widehat P$。将 $P$ 和 $\widehat P$ 分别通过同一个基础教师检测头，比较其分类与回归输出：

$$
\mathbf z(P)=H_s^{T,\mathrm{cls}}(P),
\qquad
\widehat{\mathbf d}(P)=H_s^{T,\mathrm{reg}}(P),
$$

$$
\mathbf z(\widehat P)=H_s^{T,\mathrm{cls}}(\widehat P),
\qquad
\widehat{\mathbf d}(\widehat P)=H_s^{T,\mathrm{reg}}(\widehat P).
$$

至少报告以下偏差：

$$
E_{\mathrm{feat}}
=
\mathbb E\left[
\frac{\|\widehat p-p\|_2}{\|p\|_2+\varepsilon}
\right],
$$

$$
E_{\mathrm{cls}}
=
\mathbb E\left[
\left\|
\sigma(\mathbf z(\widehat P))-\sigma(\mathbf z(P))
\right\|_1
\right],
$$

$$
E_{\mathrm{reg}}
=
\mathbb E\left[
\operatorname{SmoothL1}
\left(\widehat{\mathbf d}(\widehat P),\widehat{\mathbf d}(P)\right)
\right].
$$

其中，$E_{\mathrm{cls}}$ 和 $E_{\mathrm{reg}}$ 是直接决定回放监督是否可信的核心指标，$E_{\mathrm{feat}}$ 用于辅助解释输出偏差的来源。应按检测尺度、PCA 维度和类别分别统计，不能只报告一个总体重建误差。

若 PCA 重建导致明显的分类或回归输出偏差，并进一步损害回放蒸馏效果，则回退到最基础的无压缩方案：直接在展开后的原始局部特征空间中进行聚类、建立多中心局部分布并采样。此时有：

$$
\widetilde p\sim\mathcal P_{c,m}^{\mathrm{raw}},
\qquad
\widetilde P=\operatorname{reshape}(\widetilde p),
$$

随后仍将 $\widetilde P$ 通过教师和学生检测头，使用相同的分类与回归回放蒸馏。该回退方案不执行 PCA 投影或逆变换，也不需要保存 $B_s$，但原型存储和采样成本更高。

### 8.6 全类别分类回放蒸馏

设回放 patch 集合为 $\mathcal R$，所有通过当前特征表示方案生成的回放 patch 均参与训练，分类回放损失为：

$$
\boxed{
\mathcal L_{\mathrm{PRD}}^{\mathrm{cls}}
=
\frac{
\sum_{\widetilde P\in\mathcal R}
\sum_{c\in\mathcal V_{\mathrm{all}}}
D_{\mathrm{BKL}}
\left(\widetilde p_c^T\middle\|\widetilde p_c^S\right)
}{
|\mathcal V_{\mathrm{all}}|\,|\mathcal R|+\varepsilon
}.
}
$$

对于旧类别，该项保留教师软响应。对于新类别，由于 $\widetilde p_c^T=0$，有：

$$
D_{\mathrm{BKL}}(0\|\widetilde p_c^S)
=
-\log(1-\widetilde p_c^S),
$$

其与 $\operatorname{BCE}(\widetilde p_c^S,0)$ 等价。因此，新类通道仍受到背景约束，但该约束现在是全类别回放蒸馏的一部分，而不再是独立的新类负样本损失。

### 8.7 回归回放蒸馏

同时约束学生的回归输出接近教师，所有回放 patch 等权参与：

$$
\boxed{
\mathcal L_{\mathrm{PRD}}^{\mathrm{reg}}
=
\frac{
\sum_{\widetilde P\in\mathcal R}
\operatorname{SmoothL1}
\left(\widehat{\mathbf d}^S,\widehat{\mathbf d}^T\right)
}{
|\mathcal R|+\varepsilon
}.
}
$$

完整回放蒸馏损失为：

$$
\boxed{
\mathcal L_{\mathrm{PRD}}
=
\mathcal L_{\mathrm{PRD}}^{\mathrm{cls}}
+
\lambda_{\mathrm{PRD-reg}}
\mathcal L_{\mathrm{PRD}}^{\mathrm{reg}}.
}
$$

该模块同时承担：

- 旧类分类软响应保持；
- 新类背景响应校准；
- 旧类前景定位能力保持。

## 9. ESPReg 表示稳定正则

### 9.1 历史层输入统计

对于骨干和颈部网络的第 $l$ 层，记其输入为 $x_l$，计算历史非中心二阶矩：

$$
\boxed{
\Sigma_l
=
\mathbb E_{x\sim\mathcal D_{1:t-1}}
[x_lx_l^\top].
}
$$

其特征分解为：

$$
\Sigma_l=U_l\Lambda_lU_l^\top,
$$

其中：

$$
U_l=[u_{l,1},\ldots,u_{l,d_l}],
\qquad
\Lambda_l=\operatorname{diag}
(\lambda_{l,1},\ldots,\lambda_{l,d_l}).
$$

### 9.2 累计参数偏差与局部响应扰动

定义相对于上一阶段模型的累计参数偏差：

$$
\Delta W_l=W_l^{(t)}-W_l^{(t-1)}.
$$

在固定当前层输入的局部条件下：

$$
\mathbb E\left[\|\Delta W_lx_l\|_2^2\right]
=
\operatorname{Tr}
\left(\Delta W_l\Sigma_l\Delta W_l^\top\right)
=
\sum_k\lambda_{l,k}\|\Delta W_lu_{l,k}\|_2^2.
$$

因此，历史高特征值方向通常对应更敏感的局部响应扰动。

### 9.3 可切换的特征值加权与正则损失

为区分原始二阶扰动目标与归一化、截断式启发式加权，引入权重模式开关：

$$
\varrho\in\{\mathrm{raw},\mathrm{NT}\},
$$

其中 $\mathrm{raw}$ 表示直接使用原始特征值，不进行归一化和截断；$\mathrm{NT}$ 表示先按特征值谱肘点归一化，再将权重截断至 1。统一定义：

$$
\boxed{
\omega_{l,k}^{(\varrho)}
=
\begin{cases}
\lambda_{l,k},
& \varrho=\mathrm{raw},\\[4pt]
\lambda_{l,k}^*,
& \varrho=\mathrm{NT},
\end{cases}
}
$$

其中：

$$
\boxed{
\lambda_{l,k}^*
=
\min\left(
\frac{\lambda_{l,k}}{\lambda_{l,r}},1
\right),
}
$$

$\lambda_{l,r}$ 为第 $l$ 层特征值谱的肘点特征值。因此，$\mathrm{NT}$ 模式就是此前的归一化加截断版本，$\mathrm{raw}$ 模式则保留全部原始特征值及其相对量级。

两种模式共用同一个 ESPReg 损失接口：

$$
\boxed{
\mathcal L_{\mathrm{ESP}}^{(\varrho)}
=
\sum_{l\in\mathcal L_{\mathrm{BN}}}
\alpha_l
\sum_k\omega_{l,k}^{(\varrho)}
\left\|\Delta W_lu_{l,k}\right\|_2^2,
}
$$

其中 $\mathcal L_{\mathrm{BN}}$ 表示被约束的骨干和颈部网络层，$\alpha_l$ 用于平衡不同层的损失尺度。除特别说明外，后文将当前开关对应的损失简记为 $\mathcal L_{\mathrm{ESP}}$。

矩阵形式为：

$$
\mathcal L_{\mathrm{ESP}}^{(\varrho)}
=
\sum_l\alpha_l
\operatorname{Tr}
\left(
\Delta W_lU_l
\operatorname{diag}(\boldsymbol\omega_l^{(\varrho)})
U_l^\top\Delta W_l^\top
\right).
$$

当 $\varrho=\mathrm{raw}$ 时，若令 $\alpha_l=1$，则有：

$$
\mathcal L_{\mathrm{ESP}}^{(\mathrm{raw})}
=
\sum_l
\operatorname{Tr}
\left(\Delta W_l\Sigma_l\Delta W_l^\top\right)
=
\sum_l
\mathbb E\left[\|\Delta W_lx_l\|_2^2\right],
$$

因而该模式与第 9.2 节的局部响应扰动恒等式直接对应。$\mathrm{NT}$ 模式则是有界的启发式重加权，不应再被表述为与该恒等式严格等价。

实现时，开关只改变特征值权重 $\omega_{l,k}^{(\varrho)}$，不改变历史统计、特征方向、累计参数偏差或被约束网络层。由于两种模式的损失量级不同，比较时可以固定网络结构和其余训练设置，但应分别为两种模式校准 $\lambda_{\mathrm{ESP}}$；$\alpha_l$ 仅作为层间损失平衡系数，不构成对单层内部特征值的额外归一化。

ESPReg 直接约束阶段累计参数偏差，而不是仅修改当前一步梯度。其理论解释限于固定层输入条件下的局部响应扰动，不宣称完整检测网络的严格稳定性保证。

## 10. 总训练目标

完整训练损失为：

$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{total}}
={}&
\mathcal L_{\mathrm{det}}
+
\lambda_{\mathrm{KD}}
\left(
\mathcal L_{\mathrm{cls}}^F
+
\eta_B\mathcal L_{\mathrm{cls}}^B
+
\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}^{\mathrm{KD}}
\right)\\
&+
\lambda_{\mathrm{PRD}}
\left(
\mathcal L_{\mathrm{PRD}}^{\mathrm{cls}}
+
\lambda_{\mathrm{PRD-reg}}
\mathcal L_{\mathrm{PRD}}^{\mathrm{reg}}
\right)
+
\lambda_{\mathrm{ESP}}\mathcal L_{\mathrm{ESP}}.
\end{aligned}
}
$$

各项作用为：

- $\mathcal L_{\mathrm{det}}$：当前 YOLO 检测损失，负责学习当前新类别；
- $\mathcal L_{\mathrm{cls}}^F$：真实增量图像上的可靠旧类前景分类蒸馏；
- $\mathcal L_{\mathrm{cls}}^B$：真实增量图像上的可靠旧类背景分类蒸馏；
- $\mathcal L_{\mathrm{reg}}^{\mathrm{KD}}$：真实增量图像上的可靠旧类前景回归蒸馏；
- $\mathcal L_{\mathrm{PRD}}^{\mathrm{cls}}$：历史前景 patch 上的全类别分类回放蒸馏；
- $\mathcal L_{\mathrm{PRD}}^{\mathrm{reg}}$：历史前景 patch 上的回归回放蒸馏；
- $\mathcal L_{\mathrm{ESP}}$：历史表示稳定正则。

## 11. 完整训练流程

### 11.1 基础阶段

1. 在 $\mathcal D_1,\mathcal V_1$ 上训练基础模型 $m_1$。
2. 根据 GT 和标签分配器，为每个实例选择各有效尺度上的代表性正样本位置。
3. 按检测头感受野提取检测头输入局部 patch。
4. 在每个尺度上，使用全部基础类别 patch 建立一组共享 PCA。
5. 将各类别实例 patch 投影到固定 PCA 空间。
6. 使用自动聚类为每类、每尺度建立多中心原型和局部协方差。
7. 统计骨干和颈部各层的 ESPReg 历史二阶矩及特征方向。

### 11.2 每个增量阶段

1. 从 $m_{t-1}$ 初始化 $m_t$，扩展当前新类输出通道。
2. 使用当前图像计算标准检测损失 $\mathcal L_{\mathrm{det}}$。
3. 教师在当前图像上产生旧类分类和回归响应。
4. 对候选旧类前景提取局部 patch，并通过原型匹配计算前景可靠性。
5. 对候选旧类背景单独计算背景权重，不使用前景原型匹配。
6. 分别计算前景分类、背景分类和前景回归蒸馏。
7. 根据第 8.5 节的验证结果选择回放表示：PCA 可行时，从历史旧类低维局部原型分布中采样并通过固定 PCA 重建 patch；不可行时，直接在展开后的原始局部特征空间中采样并恢复空间形状。
8. 将重建 patch 通过教师检测头，获得旧类分类和回归输出。
9. 将教师旧类概率响应与人工设置的新类零概率目标拼接为全类别教师监督。
10. 将同一重建 patch 通过学生检测头，计算全类别分类回放和回归回放蒸馏。
11. 根据相对于上一阶段模型的累计参数偏差计算 ESPReg。
12. 联合优化总损失。
13. 当前阶段结束后，提取新增类别实例 patch，使用固定 PCA 投影并建立新增类别的多中心原型。
14. 将新增类别原型加入历史原型库，并更新后续阶段所需的历史层输入统计。

## 12. 多阶段存储与推理开销

### 12.1 固定共享 PCA

若有 $S$ 个检测尺度，整个训练过程只保存：

$$
\boxed{S\text{ 组固定共享 PCA 参数}}
$$

每组包含 $\bar p_s$ 和 $B_s$。后续新增类别不增加 PCA 组数。

### 12.2 多中心局部原型

设第 $t$ 阶段累计类别数为 $C_t$，类别 $c$ 在尺度 $s$ 上形成 $M_{c,s}$ 个中心，则累计局部成分数为：

$$
N_{\mathrm{comp}}^{(t)}
=
\sum_{s=1}^{S}\sum_{c=1}^{C_t}M_{c,s}.
$$

第 $t$ 阶段新增局部成分数为：

$$
\Delta N_{\mathrm{comp}}^{(t)}
=
\sum_{s=1}^{S}\sum_{c\in\mathcal V_t}M_{c,s}.
$$

采用对角协方差时，每个局部成分只需保存一个均值向量和一个方差向量，不需要为每个成分保存一套 PCA 特征向量。

### 12.3 ESPReg 存储

ESPReg 的历史层统计和特征方向属于另一套独立参数，不计入共享 PCA 和类别原型数量。

### 12.4 推理阶段

推理时只保留学生检测器：

$$
\boxed{
\text{教师、原型库、PCA、回放采样和 ESPReg 均不参与推理。}
}
$$

因此，上述模块不改变学生模型的推理计算图，其额外成本主要发生在训练阶段。

## 13. 关键设计边界

1. **原型是检测头输入 patch，而不是最终分类层前的单向量。** patch 半径由检测头中心输出的空间感受野决定。
2. **原型匹配只用于教师前景可靠性评估。** 背景位置不使用前景原型相似度，也不使用师生框 IoU。
3. **背景只参与旧类分类蒸馏。** 背景位置的教师回归输出不作为回归蒸馏目标。
4. **前景与背景分别归一化。** 防止大量背景位置主导真实图像上的蒸馏损失。
5. **回放监督来自教师输出，而不是直接将原型作为新类负样本。** 新类零概率目标只是完整回放教师监督的一部分。
6. **新类 logit 不能直接设为 0。** 概率域设为目标概率 0，或在 logit 域设为 $-M$。
7. **回放样本不使用教师置信度筛选或加权。** 采用 PCA 时，默认所有重建 patch 均可参与回放；实验中必须比较原始局部 patch 与 PCA 重建 patch 的分类、回归输出偏差。若该近似不可行，则回退到直接在展开后的原始局部特征空间中建模和采样。
8. **固定 PCA 是基础版本的简化。** 它不保证覆盖未来类别的重要方向，本版本不实现扩张和重编码；若输出偏差验证不通过，则使用无压缩的原始局部特征方案。
9. **局部高斯只是各子类簇内的近似。** 多中心结构用于缓解单中心高斯无法描述类别多模态的问题。
10. **ESPReg 提供局部解释，而非全网严格保证。** 其推导以固定当前层输入为条件。
11. **ESPReg 的特征值加权可切换。** $\mathrm{raw}$ 使用原始、未归一化且不截断的特征值，直接对应局部响应扰动恒等式；$\mathrm{NT}$ 使用归一化加截断权重，属于有界启发式重加权。两种模式应分别校准正则强度。
12. **当前直接回归检测头采用 SmoothL1 回归蒸馏。** 不套用针对 DFL 输出的分布交叉熵。
13. **稀疏一对一输出需要先匹配再蒸馏。** 不能默认教师和学生按预测索引天然对齐。

## 14. 建议的核心验证指标与消融

### 14.1 直接对应动机的错误指标

除 Old、New、All mAP 外，应分别测量：

- 基础模型在增量训练图像上，将当前新类 GT 高置信误判为旧类的比例；
- 增量模型在历史测试图像上，将旧类 GT 高置信误判为新类的比例；
- 原始局部 patch 与 PCA 重建 patch 的特征重建误差、分类输出偏差和回归输出偏差，以及这些偏差随检测尺度和 PCA 维度的变化；
- 当 PCA 近似不可行时，无压缩原始局部特征回退方案的回放效果与存储开销；
- 原型层及检测头输入特征在增量训练前后的漂移。

### 14.2 关键消融

建议至少包含：

1. 无原型匹配，仅使用教师置信度的 YOLO-LwF；
2. 前景原型匹配，但不区分前景与背景权重；
3. 仅分类回放蒸馏，不使用回归回放蒸馏；
4. 分类与回归全类别回放蒸馏；
5. 单中心原型与自动多中心原型；
6. 单向量原型与检测头输入 patch 原型；
7. 不使用 ESPReg、仅对骨干使用 ESPReg、对骨干和颈部共同使用 ESPReg；
8. 固定其余设置，分别使用 $\mathrm{raw}$ 原始特征值加权与 $\mathrm{NT}$ 归一化加截断加权，并分别校准正则强度；
9. 完整方法与各模块分别移除后的结果。

## 15. 方法总结

当前方案的完整逻辑为：

$$
\boxed{
\begin{aligned}
&\text{以 ESPReg 稳定检测头输入特征空间，}\\
&\text{以局部 patch 原型匹配筛选教师旧类前景，}\\
&\text{以前景／背景分离的 YOLO-LwF 保留真实图像上的旧类知识，}\\
&\text{以历史前景 patch 的全类别回放蒸馏保持旧类分类与定位响应，}\\
&\text{并通过新类零概率目标抑制新类边界向旧类特征区域的不合理扩张。}
\end{aligned}
}
$$

它同时处理基础模型旧类知识向新域、以及增量模型新类知识向旧域的两类错误泛化，并在不改变学生推理结构的前提下完成训练。
