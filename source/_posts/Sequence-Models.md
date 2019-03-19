---
title: 序列模型
categories:
  - 深度学习
  - NLP
mathjax: true
comments: true
tags:
  - Coursera
  - 深度学习
  - 机器学习
  - LSTM
  - GRU
  - RNN
  - Attention
  - Word Embedding
abbrlink: 94c569ba
date: 2018-12-21 10:48:09
---

deeplearning.ai的第五课:Sequence Models.讲解了如基本的RNN网络,基本的循环单元到GRU,LSTM,再到双向RNN,还有深层版的模型.常用词嵌入的特性,不同词嵌入训练方法,集束搜索和Attention模型.
<!-- more -->

## Recurrent Neural Networks
### Notation
$X^{(i)<t\>}$表示第i个训练样本的第t个输入元素
$Y^{(i)<t\>}$表示第i个训练样本的第t个输出元素
$T_{x}^{(i)}$表示第i个训练样本的输入序列长度
$T_{y}^{(i)}$表示第i个训练样本的输出序列长度

### Recurrent Neural Network Model
标准神经网络的问题:
1. 输入输出长度可能不一致
2. 不能很好共享文本不同位置学习到的特征

<center>
    <img src="/uploads/images/RNN Forward Propagation.png" title='RNN Forward Propagation' width="600">
</center>

### Backpropagation through time

<center>
    <img src="/uploads/images/Forward propagation and backpropagation.png" title='Forward propagation and backpropagation' width="600">
</center>

<center>
    <img src="/uploads/images/rnn cell backprop.png" title='rnn cell backprop' width="600">
</center>
 
### Different types of RNNs
1. 多输入多输出模型(翻译模型)
2. 多输入单输出模型(情感分析)
3. 单输入多输出模型(音乐生成)
4. 单输出单输出模型(简单神经网络)

<center>
    <img src="/uploads/images/RNN architectures.png" title='RNN architectures' width="600">
</center>

### Language model and sequence generation

<center>
    <img src="/uploads/images/RNN model.png" title='RNN model' width="600">
</center>

### Sampling novel sequences

<center>
    <img src="/uploads/images/Sampling a sequence from a trained RNN.png" title='Sampling a sequence from a trained RNN' width="600">
</center>

### Vanishing gradients with RNNs
反向传播因为同样的梯度消失的问题,后面层的输出误差很难影响前面层的计算.不管输出是什么,不管是对的,还是错的,这个区域都很难反向传播到序列的前面部分,也因此网络很难调整序列前面的计算.如果不管的话,RNN会不擅长处理长期依赖的问题.
梯度爆炸很容易发现,因为参数会大到崩溃,你会看到很多NaN,或者不是数字的情况,这意味着你的网络计算出现了数值溢出.

### Gated Recurrent Unit (GRU)

<center>
    <img src="/uploads/images/RNN unit.png" title='RNN unit' width="600">
</center>

<center>
    <img src="/uploads/images/RNNunit.png" title='RNNunit' width="600">
</center>

<center>
    <img src="/uploads/images/GRU(simplified).png" title='GRU(simplified)' width="600">
</center>

<center>
    <img src="/uploads/images/fullGRU.png" title='fullGRU' width="600">
</center>

当$\Gamma_{u}$很接近0,可能是0.000001或者更小,这就不会有梯度消失的问题了.因为$\Gamma_{u}$很接近0,这就是说$c^{t}$几乎就等于$c^{t-1}$,而且$c^{t}$的值也很好地被维持了,即使经过很多很多的时间步.这就是缓解梯度消失问题的关键,因此允许神经网络运行在非常庞大的依赖词上

### Long Short Term Memory (LSTM)

<center>
    <img src="/uploads/images/LSTM.png" title='LSTM' width="600">
</center>

> 最后公式应该为$a^{t\}=\Gamma_{o}*tanh(c^{t})$

红线显示了只要你正确地设置了遗忘门和更新门,LSTM是相当容易把$c^{<0\>}$的值一直往下传递到右边,比如$c^{<3\>} = c^{<0\>}$.这就是为什么LSTM和GRU非常擅长于长时间记忆某个值,对于存在记忆细胞中的某个值.

### Deep RNNs

<center>
    <img src="/uploads/images/deepRNN.png" title='deepRNN' width="600">
</center>

对于RNN来说,有三层就已经不少了,不像卷积神经网络一样有大量的隐含层.或者每一个上面堆叠循环层,然后换成一些深的层,这些层并不水平连接,只是一个深层的网络.基本单元可以是最简单的RNN模型,也可以是GRU单元或者LSTM单元,并且,你也可以构建深层的双向RNN网络.

## 自然语言处理与词嵌入

### 词汇表征

One-hot向量表征的一大缺点是把每个词孤立起来(内积均为0),稀疏,泛化能力不强.词嵌入(Word Embedding)则可以学习到俩个词语相似之处.

<center>
    <img src="/uploads/images/Word Embedding.png" title='Word Embedding' width="600">
</center>

### 使用词嵌入

词嵌入迁移学习:
1. 从大量文本中学习词嵌入(1-100B words) or 下载预训练好的词嵌入模型
2. 用词嵌入模型迁移到新的只有少量标注训练集的任务中(100k words)
3. 可选: 继续微调(finetune)词嵌入(通常是数据集2比较大)

注:语言模型和机器翻译使用词嵌入较少,因为这俩者数据集都较大

### 词嵌入的特性

词嵌入的一个显著成果就是,可学习的类比关系的一般性.举个例子,它能学会man对于woman相当于boy对于girl,因为man和woman之间和boy和girl之间的向量差在gender(性别)这一维都是一样的。

<center>
    <img src="/uploads/images/Analogies using word vectors.png" title='Analogies using word vectors' width="600">
</center>

### 嵌入矩阵

<center>
    <img src="/uploads/images/Embending matrix.png" title='Embending matrix' width="600">
</center>

### 学习词嵌入

<center>
    <img src="/uploads/images/Neural language model.png" title='Neural language model' width="600">
</center>

<center>
    <img src="/uploads/images/Other context-target pairs.png" title='Other context-target pairs' width="600">
</center>

研究发现,如果你想建立一个语言模型,用目标词的前几个单词作为上下文是常见做法.但如果目标是学习词嵌入,那么用这些其他类型的上下文,也能得到很好的词嵌入。

### Word2Vec

句子:'I want a glass of orange juice to go along with my cereal.'
Skip-Gram模型: 抽取上下文和目标词配对,构造一个监督学习问题.随机选一个词作为上下文词,比如选orange这个词,然后随机在一定词距内选另一个词,比如在上下文词前后5或10个词范围内选择目标词.

<center>
    <img src="/uploads/images/Word2Vec.png" title='词嵌入的简化模型和神经网络' width="600">
</center>

关键是个softmax单元.矩阵$E$会有很多参数,所以矩阵$E$有对应所有嵌入向量$e_{c}$的参数,softmax单元也有$\theta_{t}$的参数.优化这些参数的损失函数,就会得到一个较好的嵌入向量集,这个就叫做Skip-Gram模型.它把一个像orange这样的词作为输入,并预测这个输入词从左数或从右数的某个词是什么词.

算法首要的问题就是计算速度.在softmax模型中,每次需对词汇表中的所有词做求和计算.同论文提出的还有CBOW模型.

### 负采样

<center>
    <img src="/uploads/images/Negative Sampling.png" title='Negative Sampling' width="600">
</center>

生成数据的方式是选择一个上下文词(orange),再选一个目标词(juice),这就是表的第一行,它给了一个正样本并给定标签为1.然后给定$K$次,用相同的上下文词,再从字典中选取随机的词(king,book,the,of)等,并标记0,这些就会成为负样本.如果从字典中随机选到的词,正好出现在了词距内,比如说在上下文词orange正负10个词之内也没太大关系.**算法就是要分辨这两种不同的采样方式,这就是如何生成训练集的方法.**

小数据集的话,$K$从5到20比较好.如果数据集很大,$K$就选的小一点.

模型基于逻辑回归模型,不同的是将一个sigmoid函数作用于$\theta_{t}^{T}e_{c}$,参数和之前一样.这可看做二分类逻辑回归分类器,但并不是每次迭代都训练全部10,000个词,只训练其中的5个(部分选出的词K+1个)

采样负样本方法:$P\left( w_{i} \right) = \frac{f\left( w_{i} \right)^{\frac{3}{4}}}{\sum_{j = 1}^{10,000}{f\left( w_{j} \right)^{\frac{3}{4}}}}$

### GloVe 词向量

<center>
    <img src="/uploads/images/Glove Model.png" title='Glove Model' width="600">
</center>

GloVe算法做的就是使上下文和目标词关系开始明确化.$X_{ij}$是单词$i$在单词$j$上下文中出现的次数,那么这里$i$和$j$就和$t$和$c$的功能一样.若上下文指左右几个词,则会得出$X_{ij}$等于$X_{ji}$这个结论.其他时候大致相等.加权因子$f\left(X_{ij}\right)$就可以是一个函数,$X_{ij}$为0是为0(启发性方法见GloVe算法论文).$\theta_{i}$和$e_{j}$是对称的,而不像之前了解的模型,$\theta$和$e$功能不一样,因此最后结果可以取平均$e_{w}^{(final)}= \frac{e_{w} +\theta_{w}}{2}$.

GloVe差距最小化处理
$$\text{mini}\text{mize}\sum_{i = 1}^{10,000}{\sum_{j = 1}^{10,000}{f\left( X_{ij} \right)\left( \theta_{i}^{T}e_{j} + b_{i} + b_{j}^{'} - logX_{ij} \right)^{2}}}$$

两个单词之间有多少联系,$t$和$c$之间有多紧密,$i$和$j$之间联系程度如何,换句话说就是他们同时出现的频率是多少,这是由这个$X_{ij}$影响的.然后梯度下降来最小化

<center>
    <img src="/uploads/images/featurization view of word embeddings.png" title='featurization view of word embeddings' width="600">
</center>

$$\left( A\theta_{i} \right)^{T}\left( A^{- T}e_{j} \right) = \theta_{i}^{T}A^{T}A^{- T}e_{j} = \theta_{i}^{T}e_{j}$$
通过GloVe算法得到的(关系)特征表示可能是原特征的潜在的任意线性变换,最终还是能学习出解决类似问题的平行四边形映射.

> Word2Vec,负采样,GloVe 词向量是三种学习词向量嵌入的方法.

### 情绪分类

情感分类一个最大的挑战就是可能标记的训练集没有那么多.对于情感分类任务来说,训练集大小从10,000到100,000个单词都很常见,甚至有时会小于10,000个单词,采用了词嵌入能够带来更好的效果,尤其是只有很小的训练集时.

<center>
    <img src="/uploads/images/Simple sentiment classification model.png" title='Simple sentiment classification model' width="600">
</center>

该算法实际上会把所有单词的意思给平均.问题就是没考虑词序."Completely lacking in good taste, good service, and good ambiance.",忽略词序,仅仅把所有单词的词嵌入加起来或者平均下来,分类器很可能认为这是一个好的评论.

<center>
    <img src="/uploads/images/RNN sentiment classification.png" title='RNN sentiment classification' width="600">
</center>

### 词嵌入除偏

根据训练模型所使用的文本,词嵌入能够反映出性别、种族、年龄、性取向等其他方面的偏见,如Man对应Computer Programmer,那么Woman会对应?输出是Homemaker.

<center>
    <img src="/uploads/images/bias in word embedding.png" title='bias in word embedding' width="600">
</center>

1. 偏差求平均
2. 中和.对于那些定义不确切的词可以将其处理一下,避免偏见.如doctor和babysitter想使之在性别方面是中立的,而girl、boy定义本身就含有性别
3. 均衡步.防止又引入其他偏差.

论文作者训练一个分类器尝试解决哪些词是中立的.

## 序列模型和注意力机制

### 基础模型

1. 机器翻译到语音识别:seq2seq模型(Encoder-Decoder结构)
2. 集束搜索(Beam search)和注意力模型(Attention Model)
3. 音频模型

### 选择最可能的句子

<center>
    <img src="/uploads/images/Machine translation.png" title='Machine translation' width="600">
</center>

机器翻译模型可以看作是条件语言模型,因为语言模型总是全0输入,随机地生成句子,机器翻译模型需要找到最可能的翻译,提供不同的输入(Encoder),目的是选择使句子出现可能性最大(Decoder),选择方法如Beam search,为什么不用贪心每次选择概率最大的一个词呢?这并不是最佳选择.

### 集束搜索

<center>
    <img src="/uploads/images/Beam search algorithm.png" title='Beam search algorithm' width="600">
</center>

<center>
    <img src="/uploads/images/Beam search(B=3).png" title='Beam search(B=3)' width="600">
</center>

当B=3时表示每次只考虑三个可能结果,B=1即为贪心

1. 在第一次词位置选出最可能的三个单词$y^{<1\>}$
2. 在第一步基础上计算最可能的三个单词对$P(y^{<1\>},y^{<2\>}|x)$
3. 继续增加下一个单词重复上述步骤

### 改进集束搜索

最大化 
$P(y^{< 1 \>}\ldots y^{< T_{y}\>}|X)$=$P(y^{<1\>}|X)$\*$P(y^{< 2 \>}|X,y^{< 1 \>})$\*$P(y^{< 3 \>}|X,y^{< 1\ >},y^{< 2\>})\ldots$$P(y^{< T_{y}\ >}|X,y^{<1\ >}\ldots y^{< T_{y} - 1\ >})$

1. 改成最大化$logP(y|x)$,能防止数值下溢
2. 原公式倾向于长度短小的翻译结果,因此可以长度归一化(除$T_{y}$)

### 定向搜索的误差分析

<center>
    <img src="/uploads/images/Error analysis on beam search.png" title='Error analysis on beam search' width="600">
</center>

对结果将人工翻译和模型翻译对比,对比RNN模型出错率和集束搜索出错率,优化

### Bleu 得分

<center>
    <img src="/uploads/images/Bleu score.png" title='Bleu score' width="600">
</center>

这个例子中$p_{1}=5/7   p_{2}=4/6$,最后计算Bleu会在不同n-gram上取平均,但这样会侧重较短语句,因此会加上一个 BP(brevity penalty) 的惩罚因子.这给了机器翻译领域一个单一实数评估指标.

### Attention 模型

<center>
    <img src="/uploads/images/Attention.png" title='Attention' width="600">
</center>

<center>
    <img src="/uploads/images/Computing attention alpha.png" title='Computing attention alpha' width="600">
</center>

### 语音识别

略,没看懂

### 触发字检测

把一个音频片段计算出它的声谱图特征得到特征向量

## 参考链接
* [网易云课堂](https://mooc.study.163.com/course/2001280004#/info) 
* [Coursera Deep Learning 专项课程](https://www.coursera.org/specializations/deep-learning) 
* [吴恩达《深度学习》系列课程笔记](https://github.com/bighuang624/Andrew-Ng-Deep-Learning-notes/blob/master/docs/README.md) 