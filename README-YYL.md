# 方差缩放的权重更新投影正则化方法（Varaiance Scaled Projection Regluarization, VSPReg）

## 一、方法细节
### 1. 卷积操作的矩阵乘法表示

我们可以把卷积操作$\bm K\odot \bm F$看作是矩阵乘法$\bm W\bm X$，为了实现这一目标，可以对图像进行滑动窗口式的展开操作。

假设特征图$\bm F\in\R^{C_{in}\times H\times W}$，卷积核的形状是$\bm K\in \R^{C_{out}\times C_{in}\times h\times w}$，卷积步长是$s$，那么，我们就可以按照相同的步长，以卷积核相同尺寸，构造一个滑动窗口，去对特征图进行展开：

$$
\bm X[k]=\bm F[:,i*s-h/2:i*s+h/2,j*s-w/2:j*s+w/2]
$$

其中$k=i*H+j$。换而言之就是把卷积核在特征图每一个坐标位置处的卷积操作，转换成了矩阵乘法操作，如下图所示。

按照这种展开方式，矩阵和向量的形状分别是
$$
\bm W\in\R^{C_{out}\times C_{in}hw}\\
\bm X\in\R^{C_{in}hw\times H_{out}W_{out}}
$$

### 2. PCA

经过了1.的展开，此时卷积层的输入特征空间是一个$\R^{C_{in}hw}$空间。给定一个任务的数据集$\bm x_{img}\sim \mathcal D$，其各层的输入特征一定服从一个$\R^{C_{in}hw}$空间上的分布。我们可以计算这个分布的均值$\bm \mu$和协方差矩阵$\bm \Sigma$。进一步地，按照**PCA**方法，其协方差矩阵一定可以被单位正交分解：
$$
\bm \Sigma=\bm P\bm\Lambda\bm P^T
$$
其中，$\bm P=[\bm P_1, \bm P_2, \dots, \bm P_{C_{in}hw}]$是由单位正交向量作为列向量所构成的正交矩阵，$\bm\Lambda=\text{diag}(\lambda_1,\lambda_2,\dots,\lambda_{C_{in}hw})$是由各个方向上的分布方差构成的对角矩阵。通过这种操作，我们找到了一个对$\bm X$进行旋转变换$\bm h=\bm P^T\bm X$的方式，使得其在新的基向量组$\bm P=[\bm P_1, \bm P_2, \dots, \bm P_{C_{in}hw}]$下的坐标表示，各个分量之间不再有线性相关关系，在一定程度上实现了解耦合（<span style="color: red;">但可能仍然存在非线性相关关系</span>）。

从一个粗浅的角度去认识，我们可以认为，方差$\bm\Lambda$刻画了各个主成分方向$\bm P=[\bm P_1, \bm P_2, \dots, \bm P_{C_{in}hw}]$对于特征表示的贡献程度，方差越大的方向，表明不同样本的特征在这一方向上变化越大，对于输出的影响也越大。而相反，如果一个方向上方差极小，接近于0，则说明对于不同的样本，无论是正样本还是负样本，其特征在这一方向上基本不发生改变，仅仅是起到一个常量偏置的意义，那么这一方向对于后续输出的影响便不大。

### 3. 利用方差进行缩放的权重投影正则化
当模型进行增量训练时，假设权重的更新为$\Delta\bm W$，则它的行向量在基向量$\bm P$上的投影坐标是$\Delta\bm {WP}$，用方差的平方根$\sqrt{\bm\Lambda}$去对各个方向上的投影坐标进行缩放$\Delta\bm {WP\Lambda}$，我们就得到了权重更新值在不同方向上，按照各方向对于特征提取贡献程度加权的投影坐标。

这个投影的长度，就可以衡量权重的更新，对于服从分布$\mathcal D$的样本特征提取，有多大的影响程度。假设$\mathcal D$是历史任务，那么我们的目标就是要最小化权重更新对于特征提取的影响程度，从而避免增量学习当中常见的灾难性遗忘。因此，我们把投影的长度$\mathcal L_\text{vsp} = ||\Delta\bm {WP\Lambda}||_2$作为一个正则项，加入到模型训练的损失函数当中。

同时，我们也不能遗忘了特征分布的均值$\bm\mu$，即使最小化了$\Delta\bm W$在各个主成分方向上的缩放投影长度，其在均值方向上的投影长度如果不进行约束，也会造成特征分布的整体均值偏移。所以，我们继续在正则项里添加约束项
$$
\mathcal L_\text{vsp}=||\Delta\bm{WP\Lambda}||_2+||\Delta\bm{W\mu}||_2
$$

这就是**利用方差进行缩放后的权重投影正则化方法（Variance Scaled Projection Regularization, VSPReg）**。

## 二、代码实现

1. ultralytics/engine/vspreg.py

    1.1. PCAHooker是通过钩子函数，对模型各层的输入进行PCA并且管理计算结果的管理器类型。所谓钩子函数就是一个在模块每次前向计算的时候都会在最终阶段调用的，以模块的输入、输出和权重为参数的函数。所以可以用来在模型前向计算时获取到输入特征，从而计算PCA。
    
    1.2. VSPLoss是根据传入的主成分矩阵、方差序列和均值向量，计算方差缩放投影正则化损失函数的类型；
    
    1.3. VSPRegTrainer是继承原本的BaseTrainer的训练器类型，相较于BaseTrainer进行修改的部分通过注释标注明白。

    1.4. RealTimeMemoryMonitor用于监控PCA过程中的内存资源占用

2. ultralytics/engine/distillation.py

    2.1. YOLOv8DistillationLoss和PCAHooker类似，通过钩子函数，获取各层的输出向量然后计算蒸馏损失函数。

    2.2. DistillationTrainer同样也是继承了BaseTrainer，相关的改动同样已经使用注释标注出来。
    
3. ultralytics/models/yolo/detect/train.py

    3.1. VSPRegDetectionTrainer是对于VSPRegTrainer的继承，重写了_do_pca方法，具体来说，考虑了检测任务当中提供的ground truth标注，只提取bbox在特征图上框定的RoI区域内的特征进行PCA，而不是进行全局特征向量的PCA。

    3.2. DetectionPCAHooker对PCAHooker进行了继承，主要是重写了_get_sample_feature_indices函数，其可以支持根据ground truth标注提取RoI中的特征进行PCA

    3.3. DistillationDetectionTrainer对DistillationTrainer进行了继承，并没有增加什么新的逻辑，但是为了能够在检测任务当中被使用，需要像DetectionTrainer在BaseTrainer基础上添加一些函数。

4. tools/train_incremental.py用于执行增量训练。在这一份代码当中，增量训练的每一个阶段，是分开进行的。所以需要使用配置文件来记录每一个阶段使用的数据集、模型基础、中间文件的生成路径。

5. tools/eval_incremental.py用于测试增量训练得到的模型权重。

6. tools/create_incremental_dataset.py用于将一个完整的yolo数据集分割为多阶段的类增量数据集。

7. 其余tools目录下的代码均为工具代码，这里不一一列出了了，注释都有阐述。

8. 具体调度完整的多阶段增量学习的指令，位于configs目录下，每一个实验设置对应其中一个子目录。子目录里的train.sh和eval.sh是完整的训练脚本和测试脚本。