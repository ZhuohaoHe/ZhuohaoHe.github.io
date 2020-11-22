---
layout:     post
title:      "机器学习笔记--模型选择与正则化"
subtitle:   
date:       2020-10-18 12:00:00
author:     "HeZh"
header-img: "img/post-2020-bg.jpg"
catalog: true
tags:
    - Machine Learning
    - Regularization
---

## 模型选择与正则化 (Model Selection  &  Regularization)

### 偏差与方差 (Bias and Variance)

假设用某个函数 `h(x)` 去近似真实函数 `y(x)` ，其偏差和方差为：
$$
bias(h(x) ) = E[h(x) - y(x)]
$$

$$
var(h(x)) = E{\lbrace h(x) - E[h(x)] \rbrace}^2 = E[{h(x)}^2]- E[h(x)]^2
$$

*注：偏差与真实值相关，但方差与真实值无关，只与h(x)的分布有关*

### 欠拟合问题和过拟合问题 (Under-fitting and Over-fitting)

欠拟合是拟合的还不够好(欠), 对于训练的数据，模型还不能得到很好的结果。

过拟合就是指把学习进行的太彻底（过）, 虽然对于训练的数据，能够得到很好的分类/预测结果，但是对于新输入的数据，却不能得到和训练数据一样好的结果。

下图展示了在线性回归和分类问题中的欠拟合和过拟合：

<img src="/img/in-post/2020-11-20-Mode-Selection-and-Regularization/regression-overfitting.png" alt="regression-overfitting" style="zoom: 50%;" />

<img src="/img/in-post/2020-11-20-Mode-Selection-and-Regularization/classification-overfitting.png" alt="classification-overfitting" style="zoom:46%;" />

### 选择模型

从图中可以看出，一般而言，模型简单会导致欠拟合，模型复杂不易欠拟合，但是会容易过拟合。所以我们需要适当的选择模型，尽量训练结果好的同时也要避免过拟合。

把数据**随机**分成两部分, 
$$
数据 = 
\begin{cases}
训练集(Training\  Set): 用于模型训练 \\
验证集(Validation\ Set): 用于模型选择
\end{cases}
$$

利用训练集和验证集再结合上文提到的偏差与方差，就可来选择模型了。

1. 偏差是反映模型在训练集训练的表现和真实值之间的差距，可以看作是模型在训练集上的错误率
2. 方差是反映模型在验证集上的表现与训练集上的差距

根据偏差与方差的大小，可以将模型分为下面四类：

|              | **偏差  小**                       | **偏差  大**                   |
| ------------ | ---------------------------------- | ------------------------------ |
| **方差  小** | 训练误差大，测试误差大，训练效果好 | 训练误差大，测试误差大，欠拟合 |
| **方差  大** | 训练误差小，测试误差大，过拟合     | 同时欠拟合过拟合（没见过）     |

*注：方差小说明模型在验证集上的表现和训练集上的表现一样好/坏， 方差大说明模型在验证集上的表现还不如训练集上的表现，验证集上的表现更差。*

下图是在模型选择上常见的场景:

<img src="/img/in-post/2020-11-20-Mode-Selection-and-Regularization/model-complexity.png" alt="model-complexity" style="zoom:67%;" />

随着模型复杂度的增长，由欠拟合逐渐变为过拟合，Train_loss 和 Validation_loss 都在一开始时候下降，但是当模型复杂度增长到一定程度，Validation_loss 开始上升，而 Training_loss 还在下降，这就是发生了过拟合。

而我们选择最优模型，就是平衡 Training_loss 和 Validation_loss , 让这两者的 total_loss 将到最低。

### 解决欠拟合和过拟合问题的方法：

欠拟合：增加模型复杂度

+ 收集新的特征
+ 增加多项式组合特征

过拟合：

+ 增加数据（最有效的方法，但是往往不能实现）
+ 降低模型的复杂度
  + 减少特征（人为的特征过滤）
  + 正则化（Regularization）：可大幅降低方差，增大偏差

### 正则化

将目标函数变为：

$$
J(\theta) = L(\theta) + \lambda R(\theta)
$$

$$
\begin{cases}
L(\theta) = \frac 1 {2m} \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2 \ \ (原本的loss \ function)
\\
\\
R(\theta) = \sum^n_{j=1} {\theta_j}^2 \  \  (正则项)
\end{cases}
$$

> 为什么引入这种形式的正则项就可以解决过拟合的问题？
>
> 机器学习的过程，是一个通过修改参数 theta 来减小误差的过程，可是在减小误差的时候非线性越强的参数, 比如在 $ x^3 $  旁边的 theta  就会被修改得越多, 因为如果使用非线性强的参数就能使方程更加曲折, 也就能更好的拟合上那些分布的数据点。但是因为次数越高的参数对整个模型的影响越大，这样就很容易导致过拟合，所以需要使用正则项来限制。
>
> 或者说在经验风险最小化的基础上（也就是训练误差最小化），尽可能采用简单的模型，以此提高泛化预测精度。
>
> 参考文章：
>
> L1正则化与L2正则化 - bingo酱的文章 - 知乎 https://zhuanlan.zhihu.com/p/35356992

其中 $\lambda$ 被叫做正则化因子，通过对正则化因子的设置，可以有效的降低过拟合。

$ \lambda $ 的设置对模型训练结果的影响：随着 $ \lambda $ 的不断增大， training_loss 逐渐变大

+ 当 $ \lambda $ 过小 (可以参考 $ \lambda = 0$)，对过拟合的降低并不明显，模型和引入正则化之前较为相似。
+ 当 $ \lambda $  过大 (可以参考 $ \lambda = \inf$ ), 将会导致 $ L(\theta) $ 对 $ J(\theta) $ 的影响变得很小，模型变为欠拟合，并且梯度下降将无法让 $ J(\theta) $ 收敛

$ \lambda $ 的设置对模型验证结果的影响：随着 $\lambda$ 的不断增大， validation_loss 先变小后变大

+ 当一开始 $ \lambda $ 增大时，引入正则化开始其作用，降低了过拟合，让 validation_loss 开始降低
+ 当 $ \lambda $ 过大时， 由于正则项过大，导致模型变为欠拟合，所以 validation_loss 也会随之上升

如下图所示，

<img src="/img/in-post/2020-11-20-Mode-Selection-and-Regularization/lambda.png" alt="lambda" style="zoom:60%;" />

由于我们的目标函数 $J(\theta)$  发生了变化， 所以梯度下降求 $min_\theta J(\theta) $ 的方法也会相应的发生变化。注: $ \theta $ 不是从 0 开始，而是从 1 开始。

![gradient-descent](/img/in-post/2020-11-20-Mode-Selection-and-Regularization/gradient-descent.png)

#### 用矩阵表示正则项的引入：

$$
X =
\left[
\begin{matrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
...\\
(x^{(n)})^T
\end{matrix}
\right]
\ \ \ 
y = 
\left[
\begin{matrix}
(y^{(1)}) \\
(y^{(2)}) \\
...\\
(y^{(n)})
\end{matrix}
\right]
$$

$$
\theta = (X^TX)^{-1}X^Ty  \ 
\rightarrow \ 
\left( X^TX + \lambda
	\left[
		\begin{matrix}
		0 \ \ \ \ \ \ \ \ \ \ \ \\
		\ \  \ 1 \ \ \ \ \ \ \ \ \ \\ 
		\ \ \ \ \ \ \ ... \ \ \ \ \\
		\ \ \ \ \ \ \  \ \ \ \ \ 1 \\
		\end{matrix}
	\right]
\right)^{-1}X^Ty
$$

### 学习曲线

上文中提到增加数据量是降低过拟合的最有效的方法，但是并不是任何模型增加数据量就可以降低过拟合，降低 validation_loss 的。

下面将结合偏差大和方差大两种情况，来分析增加数据对模型的影响。

+ 模型简单，偏差大，欠拟合

  <img src="/img/in-post/2020-11-20-Mode-Selection-and-Regularization/learning-curve1.png" alt="learning-curve1" style="zoom: 60%;" />

  <img src="/img/in-post/2020-11-20-Mode-Selection-and-Regularization/more-data1.png" alt="more-data1" style="zoom:70%;" />

  当模型太过简单，即使输入大量的数据也不会让 validation_loss 有明显的提升

+ 模型复杂，方差大，过拟合

<img src="/img/in-post/2020-11-20-Mode-Selection-and-Regularization/learning-curve2.png" alt="learning-curve2" style="zoom: 60%;" />

<img src="/img/in-post/2020-11-20-Mode-Selection-and-Regularization/more-data2.png" alt="more-data2" style="zoom:70%;" />

​		当模型复杂，过拟合发生时，输入大量的数据可以有效的缓解过拟合。

> 为什么在数据量小的时候 train_loss 很小， validation_loss 很大？
>
> 当数据量很小的时候，模型是很容易过拟合的（参考数据只有一个），这时准确率就会很高，所以 loss 很小，而因为这时模型出现严重的过拟合，导致验证集中的新数据就很难在模型中得到很好的结果。

> 为什么 train_loss 随着数据量的增加，反而 loss 在增加 ?
>
> 因为增加数据量的过程，就是在不断打破模型的过拟合的过程，即不断提高模型的泛化能力。原本已经训练好的过拟合模型，在训练集中增加新的数据，就会导致 train_loss 增加。

### 模型性能评估

我们用训练集优化参数，用验证集选择模型，那么我们要如何对模型的性能进行评估呢？

这就需要引入另一个新的数据集，**测试集(Testing Set)**，测试集的作用有且仅有对模型的性能进行评估。

验证集和测试集的选择：

+ 验证集和测试集需要同分布

> 为什么验证集和测试集需要同分布？
>
> 验证集和测试集不同分布就会导致模型在验证上的效果很好，却在测试集上效果很差，会引入不必要的不确定性。例如：在训练人脸识别模型时，训练集和测试集使用的是清晰的图片，但是测试集使用的是模糊的图片，这样即使模型训练的结果很好，最终测试结果也会很差。
>
> 如果控制了验证集和测试集的同分布，就会减少这种不确定性。
>
> 例如：当验证集和测试集已经同分布之后，仍然出现验证集上效果很好但是测试集效果很差的情况，我们就可以分析出原因是在验证集上模型过拟合了，解决的方法就是去获得更多的验证集数据。
>
> 参考：
>
> 深度学习中训练集与测试集同一分布如何理解？ - 努力奔跑的小明的回答 - 知乎 https://www.zhihu.com/question/285978032/answer/565024440

+ 验证集通常有 1000~10000 个数据组成，需要足够大。
+ 测试集通常占总数据量的 30%, 在有大量数据的情况下越大越好
+ 但是并不需要过量的验证集和测试集去心境模型选择和性能评估，还是需要把大量的数据留给训练集使用

**交叉验证** (k flod cross validation)

+ 当数据规模较小的时候使用

+ 把数据随机划分为 k 等分，每次使用其中的 (k-1) 份做训练，剩下的做验证

+ 计算平均误差(方差)

  将 k 次得到的误差求平均值，最终平均误差最小的模型性能最好。

![k-cross](/img/in-post/2020-11-20-Mode-Selection-and-Regularization/k-cross.png)

### 对比 训练集，验证集，测试集

形象上来说**训练集**就像是学生的课本，学生 根据课本里的内容来掌握知识，**验证集**就像是作业，通过作业可以知道 不同学生学习情况、进步的速度快慢，而最终的**测试集**就像是考试，考的题是平常都没有见过，考察学生举一反三的能力。

> 为什么要有测试集？
>
> 训练集直接参与了模型调慘的过程，显然不能用来反映模型真实的能力，这样一些 对课本死记硬背的学生(过拟合)将会拥有最好的成绩，显然不对。同理，由于验证集参与了人工调参(超参数)的过程，也不能用来最终评判一个模型，就像刷题库的学生也不能算是学习好的学生是吧。所以要通过最终的考试(测试集)来考察一个学(模)生(型)真正的能力。
>
> 参考链接：
>
> 训练集(train)验证集(validation)测试集(test)与交叉验证法 - Ph0en1x的文章 - 知乎 https://zhuanlan.zhihu.com/p/35394638