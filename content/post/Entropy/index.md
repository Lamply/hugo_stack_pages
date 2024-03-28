---
title: 熵
date: 2023-02-24
math: true
categories: 
    - 基础原理
---
信息量为 I(x) = log(1/p(x))，其中 log 通常是 2 底，表示为 bit 单位。其中直觉在于，当信号 x 出现的概率 p(x) 越高，则含有的信息量应当越少，而 p(x) 越低，其含有的信息量应当越大。

熵为信息量 H(x) = E(I(x)) 的期望，可表示成：  
$$H(x)=\sum_ip(x_i)*log\frac{1}{p(x_i)}$$

关于一个样本集, 一般有两种分布, 预测的概率分布 q 和 真实的概率分布 p 。

## 交叉熵
为了编码样本集, 需要的平均编码长度为信息量的期望
$$H(X)=\sum_ip(x_i)*I(x_i)=\sum_ip(x_i)*log\frac{1}{p(x_i)}$$
因为不清楚真实的概率分布, 所以用模型预测的概率分布 q 对比采样的真实分布 p 来降低泛化误差。
$$H(p,q)=\sum_xp(x)*log\frac{1}{q(x)}$$
$H(p,q)$即为_交叉熵_, $H(p,q) >= H(p)$。它表示了用 q 来表示 p 时所需的信息量均值（？）。此处 x 代表 p、q 的支撑集，也就是非零的部分。

## 相对熵
用非真实分布 q 编码所需长度多出来的 bit 就是 _相对熵_, 也叫 _KL散度_ (_Kullback–Leibler divergence_)
$$D(p||q)=H(p,q)-H(p)=\sum_ip(i)*log\frac{p(i)}{q(i)}$$  
真实分布不改变的情况下，即 H(p) = Constant，最小化相对熵相当于最小化交叉熵。因为交叉熵计算更简单，而且训练集代表的 p 分布一般不变，所以在机器学习中通常使用交叉熵来作为代价函数。需要注意的是，KL 散度是不可互换的，在一些情况下这种性质会带来学习的不平衡（GAN）。

## JS 散度
$$JS(p||q)=\frac{1}{2}KL(p||\frac{p+q}{2}) + \frac{1}{2}KL(q||\frac{p+q}{2})$$  
使得 p、q 可以互换。但是如果支撑集不重叠，则 JS 散度为常量。
