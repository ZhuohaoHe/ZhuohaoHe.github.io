---
layout:     post
title:      "机器学习笔记--Logistic回归"
subtitle:   
date:       2020-10-18 12:00:00
author:     "HeZh"
header-img: "img/post-2020-bg.jpg"
catalog: true
tags:
    - Machine Learning
    - Logistic Regression
    - Classification
---

## 分类：Logistic回归

上一篇文章我们介绍了机器学习中的回归方法，那能否将回归方法直接应用于分类问题呢？

![bad result](/img/in-post/2020-10-18-Classification-Logistic-Regression/regression-bad-result.jpg)

*蓝色直线* : 在前八个数据之前，利用回归来分类，得到的分类结果还是相对准确的。

*红色直线* : 但是当第九个数据出现之后，回归直线由于这一个数据，产生了偏移，出现 $ h(x) > 1 $ 或者 $ h(x) < 0 $ 的情况，很显然分类错误。

*绿色直线* : 正确的分类结果

为了解决 *红色直线* 的错误分类，出现了 **Logistic回归** 

#### Logistic Regression

使用 **Logistic回归** 的目的是控制 $ 0 \leq h(x) \leq 1 $ ，原本使用的 $ h_\theta(x) = \theta^Tx $ 不符合要求，所以需要对 $ h_\theta(x) $ 改进，

$$
h_\theta(x) = g(\theta^Tx) = \frac{1}{1+e^{-\theta^Tx}}
$$

其中，g(z)函数被称为 logistic 函数 或 sigmoid 函数

$$
g(z) = \frac{1}{1+e^{-z}}
$$

**sigmoid 函数的性质**

$$
g'(z) = g(z)(1-g(z))
$$

**概率解释**

$h_\theta(x)$:对于输入 x ， 输出 y = 1 的可能性

$$
P(y=0|x;\theta) + P(y=0|x;\theta) = 1
$$

**分类边界**

由于函数 $ 0 \leq h_\theta(x) \leq 1 $ ,所以，

$$
y = \begin{cases}
	1, if \space h_\theta(x) > 0.5\\
	0, if \space h_\theta(x) < 0.5
\end{cases}
$$

 又有

$$
h_\theta(x) = g(\theta^Tx) = \frac{1}{1+e^{-\theta^ Tx}}
$$

根据 sigmoid 函数的曲线图像，

<img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/sigmoid.jpg" alt="sigmoid" style="zoom: 33%;" />

可以得到分类边界，

$$
\begin{cases}
	\theta^Tx > 0, if \space y = 1\\
	\theta^Tx < 0, if \space y = 0
\end{cases}
$$

例如下图，$ h_\theta(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_2) $

<img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/edge1.jpg" alt="edge1" style="zoom: 25%;" />

如果 $ -3+x_1+x_2 \geq 0 $ , 则预测 "y = 1"

或对于非线性分类边界，$ h_\theta(X) = g(\theta_0+ \theta_1x_1+\theta_2x_2+\theta_3x_1^2+\theta_4x_3^2) $

<img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/edge2.jpg" alt="edge2" style="zoom:25%;" />

如果 $ -1+x_1^2+x_2^2\geq0 $, 则预测 "y = 1"

#### 输入

+ 每个样本中的 x 都有相应的 y 对应，即样本集

  $$
  \lbrace (x^{(1)}, y^{(1)}), ..., (x^{(n)}, y^{(n)})  \rbrace
  $$

+ x 有 m 个样本， n 维特征

$$
X_{m \times (n+1)} = \left[
	\begin{matrix}
	\vec x^{(1)}\\
	...\\
	\vec x^{(n)}
	\end{matrix}
\right], 其中 \vec x^{(j)} = (1, x_1^{(j)}, ..., x_n^{(j)})
$$

+ y 有

$$
y_{m\times1} = \left[\begin{matrix}
	y^{(1)}\\
	...\\
	y^{(m)}
\end{matrix}\right]
$$

#### 损失函数

>  能否用线性回归中的平方损失函数？
>
> 不能，因为对于 $ h_\theta(x) $ 来说，平方损失函数是非凸函数，损失函数的最优选择是凸函数，所以我们需要继续寻找更好的损失函数。
>
> 什么是凸函数？
> <img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/convex-function.jpg" alt="convex function" style="zoom: 35%;" />

0-1损失函数？

$$
l(h_\theta(x), y) = \begin{cases} 
	1, if {\space} h_\theta(x) \ne y\\
	0, otherwise
\end {cases}
$$

0-1损失函数是最能准确表达概率的函数，但是由于它是离散的函数，不能够进行优化，所以选择0-1损失函数并不合适。

> 优化：通过对参数的调整，使得损失函数收敛到最小值，例如上一节的梯度下降法，但是由于函数是离散的，导数等工具都无法使用，所以不能进行优化。

用其他损失函数替代0-1损失函数？

最大似然估计！

$$
P(y=1|x;\theta) = h_\theta(x) \\
P(y=0|x;\theta) = 1-h_\theta(x)
$$

$$
p(y|x;\theta) = (h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$

损失函数，

$$
L(\theta) = p(y|X;\theta) 
=\prod^m_{i=1} p(y^{(i)}|x^{(i)};\theta)\\
=\prod^m_{i=1} (h_\theta(x))^{y^{(i)}}(1-h_\theta(x))^{1-y^{(i)}}
$$

**Logistic损失函数**

$$
l(\theta) = -logL(\theta) \\
=-\left[ \sum^m_{i=1}y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)})) \right]
$$

$$
Cost(h_\theta(x^{(i)}), y) = \begin{cases}
	-log(h_\theta(x)), \space if \space y = 1 \\
	-log(1-h_\theta(x)), \space if \space y = 0
\end{cases}
$$

通过图像来验证损失函数的结果，

*当 y = 1 时， 随着 $ h_\theta(x) $ 逐渐接近1，损失函数逐渐减小*

<img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/loss1.jpg" alt="loss1" style="zoom:30%;" />

*当 y = 0 时， 随着 $ h_\theta(x) $ 逐渐接近1，损失函数逐渐增大*

<img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/loss2.jpg" alt="loss2" style="zoom:33%;" />

#### 梯度下降

$$
J(\theta) =-\left[ \sum^m_{i=1}y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)})) \right]
$$

梯度下降模板

`repate {`

$$
\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}J(\theta)
$$

`}`

对于 logistic 回归的 $ J(\theta) $ ,求 导数得，

$$
\frac{\partial}{\partial\theta_j}J(\theta) = (h_\theta(x)-y)x_j
$$

*推导过程如下图*

<img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/derivative.jpg" alt="derivative" style="zoom: 33%;" />

> 似乎和上一节的线性回归相同？是否真的相同呢？
>
> ？？？

所以梯度下降公式可以更新为，

`repate {`

$$
\theta_j := \theta_j-\alpha (h_\theta(x)-y)x_j
$$

`}`

(注意，$ \theta $ 需要同步更新，即每次都将所有的 $ \theta $ 更新)

> 利用梯度下降验证：是否能用线性回归中的平方损失？
>
> 有，
> 
> $$
> \begin{cases} 
> J(\theta) = \frac1{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2\\
> \\
> h_\theta(x) = g(\theta^Tx) = \frac{1}{1+e^{-\theta^Tx}}\\
> \\
> g'(z) = g(z)(1-g(z))
> \end{cases}
> $$
>
> $$
> \frac{\partial}{\partial\theta_j}J(\theta)=(g(\theta^Tx)-y)g(\theta^Tx)(1-g(\theta^Tx))x_j
> $$
>
> 推导如下图，
>
> <img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/derivative.jpg" alt="derivative2" style="zoom:33%;" />
>
> 假设 "y = 0", 
> 
> $$
> if\space h_\theta(x) = 1\space {} \space(far\space form\space target) \rightarrow \frac{\partial}{\partial\theta_j}J(\theta) = 0 \\
> if\space h_\theta(x) = 0\space {} \space(close\space to\space target) \rightarrow \frac{\partial}{\partial\theta_j}J(\theta) = 0
> $$
> 
> 可以发现，当预测结果接近或者原理真实值时，梯度都为0，即并不能总是得到优解。
>
> <img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/square-error.jpg" alt="Square Error" style="zoom:50%;" />

#### Softmax Regression

在现实生活中，我们遇到的问题，往往都不是简单的二分类问题，而是更为复杂的多分类问题，而我们目前学到的知识，没有办法直接解决一个多分类问题。

虽然无法直接多分类，但是可以将多分类转换成多个二分类，即转换成多个一对多的分类，如下图。

<img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/one2many.jpg" alt="one2many" style="zoom: 40%;" />

为每类训练一个逻辑回归分类器 $ h_{\theta^i}(x) $ 用来预测 $ y = i $ 的可能性.
对于一个新输入 x , 做一个预测, 选择一个类别 i , 使得：

$$
\max_ih_{\theta^i}(x)
$$

但是由于 $ \sum h_{\theta^i}(x) \ne 1 $ ，所以概率之和不为1，并不能很明确的对概率进行比较，所以需要引入 **Softmax Regression**

$$
p(y=i|x;\theta) = h_{\theta^i}(x) = \frac{e^{(\theta^i)^T}x}{\sum_{j=1}^K{e^{(\theta^i)^T}x}}
$$

令 $ z = (\theta^i)^Tx $ 得，

$$
p(y=i|x;\theta) = h_{\theta^i}(x) = \frac {e^z} {\sum_{j=1}^K{e^z}}
$$

用对数似然估计得，

$$
L(\theta) = \sum^m_{i=1}log(p(y=i|x;\theta))\\
=\sum^m_{i=1}log(\frac {e^{z_{y^{(i)}}}} {\sum_{j=1}^K{e^{z_{y^{(i)}}}}})
$$

损失函数为，

$$
loss(\theta) = -L(\theta) = -\sum^m_{i=1}log(\frac {e^{z_{y^{(i)}}}} {\sum_{j=1}^K{e^{z_{y^{(i)}}}}})\\
=\sum^m_{i=1}\left[ log(\sum_{j=1}^K{e^{z_j}})-z_{y^{(i)}} \right]
$$

所以，单个样本 i 对应得损失为，

$$
loss_i(\theta)=log(\sum_{j=1}^K{e^{z_j}})-z_{y^{(i)}}
$$

>$ loss_i $ 的取值范围是多少？
>
>因为 $ loss_i(\theta) = -log(h_{\theta^i}(x)) $
>
>而 $ 0 \leq h_{\theta^i}(x) \leq 1 $, 所以 $ loss_i \geq 0$
>
>初始化每类的参数 $ \theta^l \approx 0 $, $ loss_i = ? $
>
>? ? ? 

**Softmax Regression 流程**

<img src="/img/in-post/2020-10-18-Classification-Logistic-Regression/softmax-regression.jpg" alt="softmax-regression" style="zoom:40%;" />

+ 利用 Softmax Regression 处理多个类，让多个类的概率和等于1

+ 利用 Cross Entropy 计算 loss 函数