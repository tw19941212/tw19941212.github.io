---
title: 结构化机器学习项目
categories:
  - 深度学习
mathjax: false
comments: true
abbrlink: 563918f4
date: 2018-12-20 16:58:52
tags:
- Coursera
- 深度学习
- 机器学习
---

deeplearning.ai的第三课:Structuring Machine Learning Projects.讲解了如何从误差角度分析改善模型,如何划分训练验证测试集,设定优化目标,偏差方差分析,数据不匹配问题,迁移学习和多任务学习,端到端学习的优缺点.
<!-- more -->

## 机器学习(ML)策略(1)
### Introduction to ML Strategy
Chain of assumptions in ML(Orthogonalization正交化):
1. 训练集上表现好?(可加大网络结构或用更好优化算法)
2. 验证集上表现好?(正则化或加大训练集)
3. 测试集上表现好?(加大验证集)
4. 真实数据上表现好?(验证集设置不正确或损失函数不正确)

<center>
    <img src="/uploads/images/Chain of assumptions in ML.png" width="600" title = 'Chain of assumptions in ML'>
</center>

### Setting up your goal
1. 单一数字评估指标 (方便比较算法优劣)
2. 满足(Satisficing)的指标的优化(optimizing)的指标
3. 选择验证/测试集分布反映未来真实数据或期望优化的数据
4. 验证/测试集划分大小
	* 测试集大小需能反映系统性能,具有高置信度
	* 验证集需足够大以评估不同方法

|training set|development set|test set|
|:-|:-|:-|
|98%|1%|1%|

5. 何时改变验证/测试集大小和评价指标
	* 在真实应用中表现不好的时候
	* 定义正确的评价指标能更好评估不同分类器
	* 优化评估指标

### Comparing to human-level performance
1. 为什么是人类水平?(若分类器比人类表现差该怎么办)
	* 获得更多标注的数据
	* 从人工错误分析:为什么人能分类正确?
	* 更好的偏差/方差分析
2. 理解人类表现
	* 人工分类误差是贝页斯误差近似
3. 提高模型表现技巧

<center>
    <img src="/uploads/images/Improving your model performance.png" title='Improving your model performance' width="600">
</center>

## 机器学习(ML)策略(2)
### Error Analysis
1. 误差分析:人工分析错误来源,统计不同错误类型占总数百分比,优先解决错误率最大
2. 深度学习算法对训练集随机误差(偶尔标记错误)具有鲁棒性,对系统误差(一直标记错误)无鲁棒性.验证集标记错误若严重影响评估算法能力,则需修正.验证集目标是帮助选择算法A & B.

{% note warning %}

* 对验证/测试集相同处理确保来自同一分布.
* 考虑检查算法分类正确的样本是否标记错误而不仅是分类错误样本是否标记错误.
* 训练集和验证/测试集或许来自稍微不同的分布.(这不会有太大影响)

{% endnote %}

3.  基本准则: 快速建立第一个系统,然后迭代优化.

### Mismatched training and dev/test set

1. 在不同划分(使dev和test来自同一分布)上训练和测试
2. 数据不匹配问题:

|Bayes optimal error|training error|training-dev error|dev error|main problem|
|:-|:--|:--|:--|:-|
|0%|1%|9%|10%|variance|
|0%|1|1.5%|10%|data mismatch|
|0%|10%|11%|12%|bias|
|0%|10%|11%|20%|bias and data mismatch|

<center>
    <img src="/uploads/images/Bias variance on mismatched training and dev test sets.png" width="600" title='Bias/variance on mismatched training and dev/test sets'>
</center>


1. 解决数据不匹配问题:收集更多像验证集的数据,或人工合成数据,但要避免从所有可能性的空间中只选了一小部分去模拟数据,造成过拟合人工合成的数据

### Learning from multiple tasks

迁移学习起作用当

* 任务A和任务B有相同的输入x(like image)
* 任务A有大量数据可供学习对比任务B
* 从A学习的低级别特征可能对B有帮助.

多任务学习起作用当:

* 在一系列任务上训练能从共享的低级别特征上收益时
* 通常:对每一任务的数据量是近似相同的
* 能训练足够大网络以至于能在所有任务上表现好

### End-to-end deep learning
优点:

* 让数据自学习
* 不依赖手工特征,组件
	
缺点:

* 需要大量数据
* 排除潜在有用手工特征,组件

应用端到端深度学习
关键问题: 你有足够的数据去学习从x映射到y的复杂性吗?

## 参考链接
* [网易云课堂](https://mooc.study.163.com/course/2001280004#/info) 
* [Coursera Deep Learning 专项课程](https://www.coursera.org/specializations/deep-learning) 
* [吴恩达《深度学习》系列课程笔记](https://github.com/bighuang624/Andrew-Ng-Deep-Learning-notes/blob/master/docs/README.md) 