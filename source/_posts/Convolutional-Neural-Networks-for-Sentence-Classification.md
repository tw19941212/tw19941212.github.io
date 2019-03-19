---
title: Convolutional Neural Networks for Sentence Classification
categories:
  - 深度学习
  - NLP
mathjax: true
comments: true
tags:
  - paper
  - TextCNN
  - CharCNN
  - 深度学习
  - 文本分类
abbrlink: f07060b1
date: 2019-01-12 10:41:26
description:
---
本文从四篇CNN用于文本分类的论文概括的介绍了什么是TextCNN,以及如何调整TextCNN的超参数.介绍了CharCNN,与TextCNN的不同之处就在于使用的是字符向量嵌入,最后介绍了CharCNN的deep版本,这在很多文本分类任务上达到了state-of-art.
<!-- more -->
## 什么是TextCNN

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)使用将词向量嵌入与CNN相结合的方式来区分文本,词向量是一个很好的特征提取器.CNN的作用是提取对预测任务有意义的子结构,利用多个不同size的kernel来提取句子中的关键信息(类似于多窗口大小的ngram),能够更好地捕捉局部相关性.MaxPool以相对输入序列的位置不变的方式选择显著特征.

### 单层CNN架构

文中指出,使用单层CNN即可获得用于文档分类的良好结果,使用词向量能提供用于自然语言处理的良好通用特征.若文本数据较大,对单词向量进行进一步的任务特定调整可以提供额外的性能提升.

<center>
    <img src="/uploads/images/TextCNN.png" title='An example of a CNN Filter and Polling Architecture for Natural Language Processing' width='600'>
</center>

* 激活函数: relu
* 卷积核大小: 2, 4, 5
* fliters: 100
* Dropout: 0.5
* L2正则化: 3
* Batch Size: 50
* 优化器: Adadelta

<center>
    <img src="/uploads/images/TextCNN-multichannel.svg" title='TextCNN-multichannel' width='600'>
</center>

论文结果表明除了随机初始化Embedding layer的外,使用预训练的word2vec初始化的效果都更加好.非静态(fine tune词向量)的比静态的效果好一些,multichannel表现亦不错(可参考论文数据).可在训练几个epoch后fine tune(一开始的时候卷积层都是随机初始化的,反向传播得到的Embedding层的梯度受到卷积层的影响,相当于噪声）

## 深入 TextCNN 超参数

[A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820) 对 TextCNN 进行文档分类所需的超参数进行了灵敏度分析,认为模型对配置很敏感.(建议阅读原文)

<center>
    <img src="/uploads/images/TextCnn.png" title='Convolutional Neural Network Architecture for Sentence Classification' width='600'>
</center>

* 预训练的word2vec和GloVe嵌入的选择因问题而异,但两者都比使用 one-hot 的单词向量表现更好,而 concatenate 俩者效果更优.
* 可以使用Grid Search不同的卷积核大小,一般若文本长度在50左右,则1-10是个不错的区间,若文本长度>100,则10-30可能更好.使用多个靠近最优卷积核大小的多卷积效果亦不错.建议先找最优卷积核大小,再尝试多卷积层.
* 激活函数建议选择tanh,relu,或者不使用激活函数.tanh比sigmoid有更好的原点对称性,relu比sigmoid有不饱和性,这可能是俩者比sigmoid好的原因.
* 特征映射(fliters)的数量可以选择100-600,dropout选择在0.0-0.5之间, L2正则化不加或弱L2正则化较好.未来若模型变得更加复杂导致过拟合,可以试着加大dropout.
*  实验比较了平均池化比最大化池化效果差很多,局部最大化策略和k-max pooling比 1-max pooling 效果稍差, 这可能是因为预测文本所在位置不重要, n-gram 比联合考虑整个句子更具有预测性.

## CharCNN

Conv用于文本分类表明ConvNets可以不需要了解语言的句法或语法结构.[Character-level Convolutional Networks for TextClassification](https://arxiv.org/abs/1509.01626)发现大数据集上的文本分类亦可不需要word级别知识.

文中几个比较有意思的点:
1. 适当的数据增强(文中使用的是替换近义词)能提高模型泛化能力
2. 不区分大小效果更好,可能原因是不区分单词含义不变,可以看成是一种正则化
3. CharCNN表明语言也可以看成是一种信号形式
4. n-gram TFIDF在小数据集(50K~)左右效果很不错
5. ChaarCNN可以学习非常规字符组合比如misspellings和emoticons(文中指出待进一步验证)

## 文本分类的深度卷积网络(VDCNN)

[Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781)是上一篇论文CharCNN的very deep版本,在多个数据集上达到state-of-art.在计算机视觉方面深度网络已经证实是成功的,因此作者尝试了深度网络用于文本分类.实验表明使用非常深的卷积神经网络模型(称为VDCNN)对分层特征学习有益.最关键的在于使用字符嵌入向量,而不是词嵌入向量.

文中结论:
1. 非常深的网络也可以很好用于小数据集
2. 深层网络减少了分类错误
3. 最大池化效果最好.比起其他复杂池化如k-max-pooling
4. 当网络越深错误率增大时,shortcut连接结构是重要的

## 总结

* TextCNN关键是使用词向量嵌入
* 单层TextCNN能在中等规模数据集上表现良好,并提供了调参依据
* 深层网络用于文本可能是未来发展方向

## 参考链接

* [Best Practices for Document Classification with Deep Learning](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/) 
* [Convolutional Neural Networks for Text Classification](http://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/) 
* [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/) 