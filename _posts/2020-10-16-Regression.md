---
layout:     post
title:      "机器学习笔记--线性回归"
subtitle:   
date:       2020-10-16 12:00:00
author:     "HeZh"
header-img: "img/post-2020-bg.jpg"
catalog: true
tags:
    - Machine Learning
    - Regression
---

## 线性回归

#### 输入特征

+ 每个样本中的 x 都有相应的 y 对应，即样本集

  $$
  \lbrace (x^{(1)}, y^{(1)}), ..., (x^{(n)}, y^{(n)})  \rbrace
  $$

+ 单个特征, m × 1 矩阵，其中 m 为样本个数

  $$
  X_{m \times 1} = 
  	\left[
   	\begin{matrix}
     		\vec x^T
    	\end{matrix}
    	\right]
  , \vec x^T = (1,x_1,x_2,...,x_m)
  $$
  
+ 多个特征, m × (n+1) 矩阵，其中 n 为特征个数，m 为样本个数

  $$
  X_{m \times (n+1)}= 
  	\left[
   	\begin{matrix}
     		{\vec x^{(1)}}^T\\
     		...\\
     		{\vec x^{(n)}}^T
    	\end{matrix}
    	\right]
  , {\vec x^{(j)}}^T = (1, x_1^{(j)},x_2^{(j)},...,x_m^{(j)})
  $$
  
	其中,
	
	$$
	x_i^{(j)} 表示第i类特征的第j个样本
	$$
	
+ 对应的y

  $$
  y_{m\times1} = \left[\begin{matrix}
  	y^{(1)}\\
  	...\\
  	y^{(m)}
  \end{matrix}\right]
  $$
  

#### 选择线性回归模型

+ 单变量，

$$
h_\theta(x)=\theta_0+\theta_1x
$$

+ 多变量，

  $$
  h_\theta(x)^{(1)} = \theta_0+\theta_1x_1^{(1)}+...+\theta_nx_n^{(1)}\\
  ...\\
  h_\theta(x)^{(n)} = \theta_0+\theta_1x_1^{(n)}+...+\theta_nx_n^{(n)}
  $$
  
+ 都可以用**矩阵**表示

  $$
  h_\theta(x) = X_{m \times (n+1)} \space \Theta_{(n+1) \times 1} 
  $$
  
  其中 Θ 矩阵为,
  
  $$
  \Theta_{(n+1) \times 1} =	
  	\left[
   	\begin{matrix}
  		\theta_0 \\
  		...\\
  		\theta_n
    	\end{matrix}
    	\right]
  $$
  

#### 损失函数

损失函数用来衡量线性模型是否和所给数据是最符合的。

损失函数应该满足的**条件**：

+ 非负：不存在负损失
+ 预测结果h(x)与真实值y的差别小，则损失小，反之损失大。即损失函数能够正确反映预测和真实结果的偏差。

在线性回归中，常用的损失函数是**平方**损失：

$$
loss(h_\theta(x^{(i)}),y^{(i)})=\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2
$$

构造出合适的损失函数主要的作用是：

+ 正确的反映出预测结果与真实值的偏差
+ 利用损失函数值不断降低为结果，寻找**优化**模型参数的方法

> 目标函数：
>
> 在损失函数上简单加工得到，用于接下来的优化操作

**三要素**：

+ 假设:

	$$
	h_\theta(x)=\theta_0+\theta_1x
	$$
	
+ 目标函数：

	$$
	J(\theta_0,\theta_1)=\frac1{2m} \sum_{i=1}^ml(h_\theta(x^{(i)}),y^{(i)})=\frac1{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2
	$$
	
+ 优化算法：给定训练集，如何找到最优的参数 θ 使得

	$$
	\min_{\theta_0,\theta_1}J(\theta_0,\theta_1)
	$$
	

#### 梯度下降法寻找最优参数

`repeat until convergence {`

$$
\theta_j:=\theta_j-\alpha\frac \partial {\partial \theta}J(\theta_0,\theta_1), for(j=0\space and \space j=1) 
$$

`}`

利用泰勒展开证明，

<img src="/img/in-post/2020-10-16-Regression/proving_process.jpg" alt="proving_process" style="zoom:50%;" />

+ **θ0和θ1要同步更新**

  <img src="/img/in-post/2020-10-16-Regression/synchronize.jpg" alt="synchronize" style="zoom:50%;" />

+ **批处理梯度下降**

  梯度下降的每一步都使用所 有的训练样本.

> **思考**
>
> 1. *能否保证找到最优的参数？*
>
>    <u>不一定，有可能进入局部最优</u>
>
>    <img src="/img/in-post/2020-10-16-Regression/local_optimum.jpg" alt="local optimum" style="zoom:33%;" />
>
> 2. 能否保证目标函数是收敛的？
>      即：梯度下降法参数更新能是否保证目标函数的值下降?
>
>      <u>不能，有可能到达鞍点，在鞍点导数为0</u>
>
>      <img src="/img/in-post/2020-10-16-Regression/saddle_point.jpg" alt="saddle point" style="zoom:33%;" />
>
> 3. 如何选择参数α(学习率)?
>
>      <u>通过不断调试，找到合适的 α 值。</u>
>
>      <u>自动收敛测试：每次迭代损失函数 J(θ) 是否减少。</u>
>
>      <u>选择收敛条件：如定义收敛为如果 J(θ) 在一次迭代中减少不超过 10^(-3)</u>
>
>      <img src="/img/in-post/2020-10-16-Regression/alpha.jpg" alt="alpha" style="zoom:50%;" />
>
>      + 对于足够小的 α，J(θ) 应该在每一次迭代中减小
>      + 如果 α 太小，梯度下降算法会收敛很慢
>      + 如果 α 太大，梯度下降算法则不会收敛：发散或震荡
>
>      <img src="/img/in-post/2020-10-16-Regression/alpha2.jpg" alt="alpha2" style="zoom: 33%;" />

#### 特征尺度归一化

确保特征在相同的尺度，保证梯度下降在每一个方向上都是均匀的。避免出现在某一个特征已经学习完毕，而另外的特征还在缓慢的学习。

归一化的方法：

+ 范围归一化

  使得每个特征尽量接近某个范围，如0<xi<1

+ 零均值归一化

  用 xi-μ 代替 xi，其中 μ 为 x 从1到m的均值

+ 零均值+范围归一化

  前两种方法的结合

+ 零均值单位方差归一化

  $$
  \frac{x_i-\mu_i}{\sigma} \rightarrow x
  $$



#### 正规方程

*除了上面介绍的采用迭代的方法以外，还有哪些方法？*

令目标函数的微分为 0 ，然后求解方程，得到目标函数的最低点位置。

用矩阵的方式表示 J(θ)

$$
X \Theta - y =
	\left[
 	\begin{matrix}
		(x^{(1)})^T\theta \\
		...\\
		(x^{(m)})^T\theta
  	\end{matrix}
  	\right] - 
  	\left[
 	\begin{matrix}
		y^{(1)} \\
		...\\
		y^{(m)}
  	\end{matrix}
  	\right] = 
  	 \left[
 	\begin{matrix}
		(x^{(1)})^T\theta - y^{(1)} \\
		...\\
		(x^{(m)})^T\theta - y^{(m)}
  	\end{matrix}
  	\right]
$$

所以，

$$
J(\theta)=\frac1{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2=\frac1{2m}(X\Theta-y)(X\Theta-y)^T
$$

解得，

$$
\theta = (X^TX)^{-1}X^Ty
$$

推导过程如下图

<img src="/img/in-post/2020-10-16-Regression/normal_equation.jpg" alt="normal equation" style="zoom:50%;" />

其中 `tr()` 为矩阵的迹

**对比 梯度下降 和 正规方程**

梯度下降：

+ 需要选择合适的 .
+ 需要多次迭代. 
+ 即使n很大，效果也很好.

正规方程：

+ 不需要选择 . 
+ 不需要迭代. 
+ 需要计算 
+ 如果n很大，速度将会很慢.

*如果矩阵不可逆怎么办？*

<u>太多的特征 (如 m <= n ), 需要删减一些特征, 或者进行正则化.</u>