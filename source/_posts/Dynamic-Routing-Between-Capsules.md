---
title: Dynamic Routing Between Capsules
categories:
  - 深度学习
  - CV
mathjax: true
comments: true
tags:
  - paper
  - CNN
  - 深度学习
  - Capsule
  - 图像处理
abbrlink: 9a4ff7b0
date: 2019-01-02 22:20:00
description:
---
Capsule是一组**vector in vector out**的神经元,每一个胶囊代表某一特定实体,如对象或对象部分.其向量模长代表实体存在概率,向量参数代表实体实例化参数.低级别Capsule向量通过变化矩阵实现对高级别Capsule实体参数进行预测.当多个预测一致时,该高级别Capsule会更加活跃,借由动态路由算法加强预测.
<!-- more -->
> 本文综合多篇中文和外文博客(原文链接均在参考链接中给出),结合自己理解转述而来,[original paper](https://arxiv.org/pdf/1710.09829.pdf) by Hinton

## CNN 的局限性
1. 需要**大量训练数据**(测试时只能识别已经训练的特征).Capsule能使用更少数据实现更好泛化
2. 不能处理**复杂场景**(如特征重叠),而Capsule能很好处理复杂场景(crowded scenes,如数字重叠),因为Capsule能很好实现特征分配(通过动态路由)
3. Pooling层丢失大量信息,降低了空间分辨率(spatial resolution),所以输出对输入的微小变化不敏感,这在需要细节信息的任务(语义分割)上是糟糕的.当然通过构建复杂CNN网络能恢复信息损失.而Capsule能精确捕捉细节信息(pose information,如角度,厚度,大小,位置等)而不是丢失后再恢复信息.因此微小输入变化导致输出变化,这被称为**等效性**(equivariance).因此Capsule能在不同的视觉任务使用相同且简单的结构.
4. CNN需要额外的组件来自动识别不同物体属于哪个对象(如这个胳膊属于这只羊),而Capsule提供了对象的**层次结构**.
5. CNN**忽略结构信息**,仅仅考虑'有没有',而没有考虑feature map的结构关系,包含位置,角度等.

## 人类视觉识别
任何物体都是由多个更小的实体组成.例如,树由树干,树冠和树根组成.这些部分形成层次结构.树冠还包括树枝和树叶.

<center>
    <img src="/uploads/images/Capsule-tree_parts_diagram.png"  title='Capsule tree parts diagram' width="600">
</center>

当我们看物体时,我们的眼睛就会形成一些**固定点(fixation points)**,这些固定点的相对位置和性质有助于我们的大脑识别这个物体.因此,我们的大脑不必处理每个细节.只要看到一些树叶和树枝,我们的大脑就会认出树冠.并且树冠在树干上.结合这种层次信息,我们的大脑知道有一棵树.从现在开始,我们将对象的各个部分称为实体.(the parts of the objects as entities)

<center>
    <img src="/uploads/images/Capsule-tree.png"  title='Capsule-tree' width="600">
</center>

## CapsNets 到底是什么?
简短的说,CapsNet由capsule(胶囊)而非神经元组成.胶囊是一小组神经元,其学习检测图像的给定区域内的特定对象(如矩形),并且输出向量(如8维向量).其**长度**表示对象存在的概率,其**方向**编码对象的姿势参数(如位置,旋转等).如果稍微改变输入对象(如移位,旋转,调整大小等),胶囊将输出相同长度的矢量,但方向略微不同.因此.胶囊具有等效性.
与常规神经网络相似,CapsNet也有多个层.最下层的胶囊称为主胶囊(primary capsules),它们中的每一个都接收图像的一小部分作为输入(receptive field),并且它试图检测特定图案的存在和姿势.更高层的胶囊(routing capsules)可以检测更大和更复杂的物体(如船只).

<center>
    <img src="/uploads/images/two-layer CapsNet.png"  title='the primary capsule layer has two maps of 5x5 capsules, while the second capsule layer has two maps of 3x3 capsules. Each capsule outputs a vector' width="600">
</center>

使用常规卷积层即可实现主要胶囊层.例如,论文中,使用两个卷积层,输出256个包含标量的6x6特征映射.重塑此输出以获得包含8维向量的32个6x6特征映射.最后,使用新颖的squash函数来确保这些向量的长度在0和1之间(表示概率).这给出了主胶囊的输出.
下一层中的胶囊也会尝试检测物体及其姿势,但工作方式却截然不同,使用称为路由协议的算法.

## 路由协议(Routing by agreement)
假设只有两个主要胶囊:一个矩形胶囊和一个三角形胶囊,并假设它们都检测实体.注意到矩形和三角形在船的姿势上是一致的,而他们对房子的姿势非常不同意.因此,矩形和三角形很可能是同一艘船的一部分.(值得注意的是实体的形状及整体/部分之间的关系是在训练中学习的)

<center>
    <img src="/uploads/images/Routing by agreement, step 1.png"  title='predict the presence and pose of objects based on the presence and pose of object parts, then look for agreement between the predictions. Image by Aurélien Géron.' width="600">
</center>

由于现在确信矩形和三角形是船的一部分,因此将矩形和三角形胶囊的输出更多地发送到船舱中是有意义的,而对于房屋舱更少:这样,船舱将接收更有用的输入信号,房屋胶囊将收到更少的噪音.即在达成一致时增加路由权重,并在出现不一致时减少路由权重.

<center>
    <img src="/uploads/images/Routing by agreement, step 2.png"  title='update the routing weights. Image by Aurélien Géron.' width="600">
</center>

路由协议算法涉及**一致性检测+路由更新**的几次迭代(注意,这发生在每次预测中,而不仅仅是训练时,不仅仅是一次).这在拥挤的场景中尤其有用.在混淆场景中很可能会收敛到一个更好的解释:底部的船,顶部的房子.模糊性被'解释了':下方的矩形最好用船的存在来解释,这也解释了下三角形,一旦解释了这两个部分,其余部分很容易被解释为房屋.

<center>
    <img src="/uploads/images/Routing by agreement can parse crowded scenes.png"  title='Routing by agreement can parse crowded scenes, such as this ambiguous image, which could be misinterpreted as an upside-down house plus some unexplained parts. Instead, the lower rectangle will be routed to the boat, and this will also pull the lower triangle into the boat as well. Once that boat is “explained away,” it’s easy to interpret the top part as a house. Image by Aurélien Géron.' width="600">
</center>

## CapsNet背后的数学
假设层$l$和$l+1$分别具有$m$和$n$个胶囊.我们的任务是在给定层$l$的激活向量下计算层$l+1$处胶囊的激活向量.设$u$表示第$l$层胶囊的激活向量.我们必须计算$v$,胶囊在$l+1$层的激活向量。
对于层$l+1$处的胶囊$j$:
1. 我们首先通过层$l$处的胶囊计算**预测向量**.胶囊$i$($l$层)对胶囊$j$($l+1$层)的预测向量由下式给出

$$\boldsymbol{\hat{\textbf{u}}}\_{j|i} = \boldsymbol{\textbf{W}}\_{ij}\boldsymbol{\textbf{u}}_{i}$$

$W_ {ij}$是权重矩阵
2. 计算胶囊$j$的**输出矢量**.胶囊$j$输出向量是胶囊层$l$capsules胶囊给出的所有预测向量的加权和

$$s_j = \sum_{i=1}^{m}{c_{ij}\boldsymbol{\hat{\textbf{u}}}_{j|i}}$$

标量$c_ {ij}$称为胶囊$i$($l$层)对胶囊$j$($l+1$层)之间的**耦合系数**.系数由**迭代动态路由**算法确定
3. 在输出向量上应用**squashing**函数来获得激活向量

<center>
    <img src="/uploads/images/Capsule-squash.png"  title='squash' width="600">
</center>

## 动态路由算法
层$l+1$的激活向量将反馈信号发送到层$l$处的胶囊.如果胶囊$j$($l+1$层)的激活向量与胶囊$i$($l$层)的预测矢量一致,则它们的点积应该比较大.因此,预测向量的'权重'在$j$的输出向量中增加.换句话说,那些贡献越大的预测向量在输出向量(激活向量)中具有更多的权重.循环持续4-5轮.(这像一种聚类算法,可以参考链接4)
低级别胶囊对高级别胶囊的的预测权重总和应该为1
$$c_{ij} = \frac{\exp(b_{ij})}{\sum_{k}{\exp(b_{ik})}}$$ 
显然
$$\sum_{k}{c_{ik}} = 1$$
logit$b_{ij}$表示胶囊$i$($l$层)和胶囊$j$($l+1$层)是否具有强耦合.换句话说,它是由胶囊$i$解释胶囊$j$的存在性大小的量度.所有$b_{ij}$初始化应该是相等的.

**Routing algorithm:**

{% note info %} 

Given: 预测向量$\boldsymbol{\hat{\textbf{u}}}\_{j|i}$, 迭代次数$r$
对胶囊$i$(层$l$)和胶囊$j$(层$l+1$):$b_{ij} = 0$
for $r$ iterations do:
&emsp; 对所有胶囊$i$(层$l$): $c\_{i }= softmax(b_i)$ **(对高级别胶囊预测权重总和为1)**
&emsp; 对所有胶囊$j$(层$l+1$): $s\_{j} = \sum\_{i=1}^{m}{c\_{ij}\boldsymbol{\hat{\textbf{u}}}\_{j|i}}$ **(输出向量是预测向量的加权和)**
&emsp; 对所有胶囊$j$(层$l+1$): $\textbf{v}\_{j} = \textbf{squash}(\textbf{s}\_{j})$ **(应用激活函数)**
&emsp; 对所有胶囊$i$(层$l$)和胶囊$j$(层$l+1$): $b_{ij}=b_{ij}+\boldsymbol{\hat{\textbf{u}}}\_{j|i} \cdot \textbf{v}_{j}$ 
返回 $\textbf{v}_j$
{% endnote %}

循环中的最后一行非常重要.这是路由发生的地方.如果乘积$\boldsymbol{\hat{\textbf{u}}}\_{j|i} \cdot \textbf{v}\_{j}$很大,它将增加$b\_{ij}$,这将增加相应的耦合系数$c\_{ij}$,这反过来将使乘积$\boldsymbol{\hat{\textbf{u}}}\_{j|i} \cdot \textbf{v}_{j}$更大.(在链接4里面博主有自己的想法)

## 路由算法优势
在CNN中,存在池化层.通常使用MaxPool,这是一种非常原始的路由机制.局部池化中最活跃的特征(比如4x4网格)被路由到更高层,而更高级别的检测器在路由中没有发言权.将其与CapsNet中引入的协议路由机制进行比较,只有那些与高级探测器一致的功能才会被路由.这是CapsNet优于CNN的优势.它具有卓越的动态路由机制(动态,因为要路由的信息是实时确定的).

## 优点和缺点

**Prons:**
1. 更少训练数据
2. 等效性保留了输入对象的位置信息
3. 动态路由算法对重叠对象(特征)很有用
4. 自动计算了输入物体的层次结构
5. 激活向量可解释性更强

**Cons:**
1. 训练慢(因为动态路由的内循环)
2. 没有在大数据集(如ImageNet)上测试
3. 在复杂数据集CIFAR10上效果不好
4. 不能区分彼此靠近的俩个相同类型的相同物体(Problem of crowding)

## 参考链接

* [Introducing capsule networks](https://www.oreilly.com/ideas/introducing-capsule-networks) 
* [Beginner's Guide to Capsule Networks](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-capsule-networks) 
* [Understanding Capsule Network Architecture](https://software.intel.com/en-us/articles/understanding-capsule-network-architecture) 
* [揭开迷雾，来一顿美味的Capsule盛宴](https://kexue.fm/archives/4819)
* [Understanding Hinton’s Capsule Networks. Part I: Intuition.](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b) 
* [GitHub:awesome-capsule-networks](https://github.com/sekwiatkowski/awesome-capsule-networks#dynamic-routing-implementations) 
* [一个注释非常详细的tensorflow源码](https://github.com/ageron/handson-ml/blob/master/extra_capsnets-cn.ipynb) 
* [CSDN:Dynamic Routing Between Capsules（NIPS2017）](https://blog.csdn.net/xyj1536214199/article/details/78698326) 