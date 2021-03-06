---
layout:     post
title:      "认识元学习"
subtitle:   
date:       2020-11-23 12:00:00
author:     "HeZh"
header-img: "img/post-2020-bg.jpg"
catalog: true
tags:
    - Machine Learning
    - Meta Learning
---

## 元学习

### 认识元学习(Meta-Learning)

#### 小样本学习(few-shot learning)

近些年来，深度学习在 CV、NLP 等领域表现的十分优秀，并且一直在不断的提高他的性能，但是这都是在有大量的数据可以供模型进行训练的基础上。例如在2015年何凯明和他在微软的团队报告说，他们的模型在对来自 ImageNet 的图像进行分类时表现优于人类，这也只能说明深度学习在利用一千万个图像来解决特定任务方面比我们人类强罢了。

那如果有些领域的数据并没有那么多，或者换句话说我们并不都像是 Google 和 Microsoft 可以获得那么多的数据，那这时该如何利用这么少的样本进行学习呢？例如我们人类，就是利用小样本学习的典型例子，我们在还是小朋友的时候并不需要学习几万或十几万张猫猫狗狗的图像才能分清邻居家的小动物是猫还是狗，而往往是在我们见到过几次小猫小狗后，就可以很容易的分清在路上遇到的是哪种宠物了。

而我们现在想要用机器也能学会这种能力，而从少数的样本中学习的这个问题，就被成为小样本学习(few-shot learning)。

#### 什么是元学习(meta-learning)

对于小样本学习而言，目前最流行的解决办法是使用元学习，又称为 learning to learn(学会学习)。什么叫做 learning to learn 呢？

+ 这里的 learn 就是机器学习中的 model 
+ learning to learn 就时利用机器自己去学习一个 model

![learning-to-learn](/img/in-post/2020-11-23-What-is-Meta-Learning/learning-to-learn.png)

在机器学习中我们需要学习的是一个解决特定问题的函数，例如判断这幅图片中有没有狗，或者判断这句话中所蕴含的情绪是什么，我们输入数据，就可以得到我们想要的分类结果或其他结果。但是对于元学习来说，他并不学习这些特定的任务，而是学习一个方法，这个方法能够学习如何解决特定的问题~~(禁止套娃)~~。



#### 定义元学习

定义函数集合 --> 定义 Loss function --> 选择最优

##### 定义函数集合(定义模型需要学习什么)

我们可以发现，现在绝大部分的机器学习模型中的许多关键步骤都是我们人为自己设计的，以深度学习的神经网络为例，下面几个地方都是需要我们人为来设置的：

+ 网络的结构
+ 初始化的参数的值
+ 更新参数的方法

![DeepLearning](/img/in-post/2020-11-23-What-is-Meta-Learning/DeepLearning.png)

所以元学习的目的之一就是尝试能不能将这些需要手工设计的部分(上图中红框的部分)，让机器自己去学习。

##### 定义 Meta-Learning 的表现好坏 (Loss function)

对于普通的机器学习，我们通过比较模型的输出和真实的 label 得到 loss ， 来评估模型的好坏，但是对于元学习，我们需要评估的不是多个数据，而是**多个任务**。我们需要输入多个任务，让模型针对不同的任务训练出对应解决任务的模型，再判断这些模型在其对应任务上的好坏，将最后的 loss 加和得到元学习模型的 loss，

$$
Loss(F) = \sum_{n=1}^Nloss^{n}
$$

其中 $ F $ 代表元学习的模型， $ loss ^n$ 代表元学习模型对应任务 n 训练处的模型的 loss 。

这里我们可以发现机器学习和元学习的训练集和测试集差别很大

$$
\begin{cases}
	\text{Mechine Learning:  } 
	\begin{cases}
		\text{Training Set:  } \ {(data,label)}
		\\
		\text{Testing Set:  } \ {(data, label)}
	\end{cases}
	\\
	\\
    \text{Meta Learning:  } \ (task)
    	\begin{cases}
    		\text{task 1: } 	
    		\begin{cases}
                \text{Support Set:  } \ {(data,label)}
                \\
                \text{Query Set:  } \ {(data, label)}
               \end{cases}
    		\\
    		\text{...}
    		\\
    		\text{task n: } 
    				\begin{cases}
                        \text{Support Set:  } \ {(data,label)}
                        \\
                        \text{Query Set:  } \ {(data, label)}
                    \end{cases}
    	\end{cases}
\end{cases}
$$

##### 选择最好模型

$$
F^* = \arg\min_F Loss(F)
$$

选择到 loss 最小的 Meta Learning 模型后，利用新的 Task 来测试他的好坏。

$$
\text{Train(New Task)}  \stackrel{F^*}{\longrightarrow} f^*\\
\text{Test(New Task)} \stackrel{f^*}{\longrightarrow} loss
$$

利用新的任务去训练模型，得到解决这个任务的模型，再用测试集去测试这个训练出的任务，判断最终的好坏。

### 元学习的几种算法

#### MAML(Model-Agnostic Meta-Learning)

面对上文提到的由人工设计的那几个步骤，MAML 主要专注于学习初始的参数(initialization parameter)。

在 MAML 中的 loss function 为：

$$
L(\phi) = \sum^N_{n=1} l^n(\hat{\theta}^n)
$$

其中, 

+ $\phi$ 为我们的最终目标的初始化参数
+ $\hat{\theta} ^n$ 是元学习对 task n 学到的参数 $\theta$ (以 $\phi$ 为初始化参数)
+ $ l(\hat{\theta} ^n) $ task n 的损失函数(参数为**训练过后**的 $\hat{\theta} ^n$) 

这里 MAML  并不在意(完全不 care)参数$ \phi $ 在 task 的表现如何，而在意用 $ \phi $ 训练出来的 $\hat{\theta} ^n$ 表现如何。即 MAML 更加在意潜力。

需要注意的是，这里的 $ \hat{\theta} ^n $ 是只训练**一步**后得到的参数，

![theta](/img/in-post/2020-11-23-What-is-Meta-Learning/theta.png)

> 为什么这里 theta 只训练一步呢？
>
> + 只训练一次时间短
> + 只训练一步就可以表现优秀，是我们的目标
> + 数据量有限制

梯度下降：

$$
\phi \leftarrow \phi - \alpha \nabla_\phi L(\phi)
$$

图解 MAML 参数的学习过程：

+ 绿色：$ \phi^0 $ 在 task m 中训练一步，得到 $ \hat \theta ^ m $ ，之后再训练一步，得到 $\phi ^ 0$ 的更新方向

+ 蓝色：$ \phi ^ 0$ 先按照绿色最后一步的方向进行更新，得到 $ \phi ^ 1 $ 

+ 橙色： $ \phi ^ 1 $ 在 task n 中训练一步，得到 $ \hat \theta ^ n $，之后再训练一步，得到   $ \phi ^ 1 $ 的更新方向

+ 蓝色： $ \phi ^ 1 $ 按照橙色最后一步的方向进行更新，得到 $ \phi ^ 2 $

  . . .

![implement](/img/in-post/2020-11-23-What-is-Meta-Learning/implement.png)



#### Reptile

Reptile 也是针对初始参数学习，但相比 MAML 要更加简单。

![Reptile](/img/in-post/2020-11-23-What-is-Meta-Learning/reptile.png)

+ 绿色：$ \phi^0 $ 在 task m 中训练(不只训练一步，而是训练结束)，得到 $ \hat \theta ^ m $ 

+ 蓝色：$ \phi^0 $ 沿着 $ \phi^0 $ 到 $ \hat \theta ^ m $ 的方向更新一步，得到 $ \phi^1 $ 

+ 橙色：$ \phi^1 $ 在 task n 中训练(不只训练一步，而是训练结束)，得到 $ \hat \theta ^ n $ 

+ 蓝色：$ \phi^1 $ 沿着 $ \phi^0 $ 到 $ \hat \theta ^ m $ 的方向更新一步，得到 $ \phi^2 $ 

  . . . 



上面提到的几种算法，都是在学习初始化的参数，那么有没有方法能够直接学习模型的结构，或者是其他的超参数( 学习率等等 )？

在上文中，我们学习了 Meta-Learning 是一种 Learning to learn 的方法，他可以让机器学会学习，而不再像传统的机器学习一样只能解决一种特定的任务，而是可以找到解决问题的方法。

我们还学习了几种利用 Meta-Learning 对模型的初始化参数进行学习的算法，但是初始化参数在模型中只是很小一部分，在这篇文章中，将介绍如何让机器参与设计模型中更多的元素。

#### 数学背景

举例，例如我们想要让机器选择一个模型中的一层，这个层由两种参数决定，一个是层深度(Layer Depth)，一个是层宽度(Layer Width)，这时就有一种很显然的方法，就是让机器去尝试所有的宽度和深度(Grid Search)，然后选择最好的一种，但是这样的代价太大，因为总的尝试次数是 (层宽度个数 * 层深度个数)，更不要说每次尝试都需要很长的时间。那么就出现了另一种方法，随机选取(Random Search)，随机选取这两种参数，并将随机选择的结果进行尝试，这样显然比全部尝试要少很多，但是这样随机的选择难道不会错过很多很好的选择嘛？

![grid-and-random](/img/in-post/2020-11-23-What-is-Meta-Learning/grid-and-random.png)

如果我们一定要选择最好的那一个结果，那么随机选取的方法确实是可能错过最好的那一个，因为毕竟是随机的。但是如果我们做这样一个假设：

> top K result are good enough

那么，随机选取的结果就可以做到速度快而且效果好。证明如下：

$$
\text{假设有 N 个点，选取一个点是前 K 个的概率是} \frac K {N} \\
\text{选取 x 次，选到前 K 个的概率为  } 1-(1-\frac K {N})^x \\
\text{如果 N = 1000， K = 10 那么 x = 230 就可以让概率  > 90%} \\
\text{如果 N = 1000，K = 100 那么 x = 22 就可以让概率  > 90%}
$$

#### AutoML

AutoML 是为了解决给定一个数据集（任务），如何找到较好的“起始算法“作为起点进行训练，或是给定一个数据集以及算法，如何找到合适的超参数。 他也是一种 Meta-Learning。

模型选择各种参数，设计一个网络，训练网络得到结果，反馈给模型，模型通过反馈的结果重新调整网络参数的设计，直到训练出一个更好的。

![train-network](/img/in-post/2020-11-23-What-is-Meta-Learning/train-network.png)

各种参数，模型的结构不同，需要设置的学习方法也不同，但核心思想相同：人工将需要学习的超参数拆分成多个部分的组合，为每个部分提供多个”组件“，模型将利用这些组建组合拼装这些超参数，得到超参数。

学习率:
![learning-rate](/img/in-post/2020-11-23-What-is-Meta-Learning/learning-rate.png)

激活函数：

![activation-function](/img/in-post/2020-11-23-What-is-Meta-Learning/activation-function.png)

> 参考资料：
>
> [Meta Learning - MAML   --Hung-yi Lee](https://www.youtube.com/watch?v=EkAqYbpCYAc)
>
> [Tuning Hyperparameters   --Hung-yi Lee](https://www.youtube.com/watch?v=c10nxBcSH14) 