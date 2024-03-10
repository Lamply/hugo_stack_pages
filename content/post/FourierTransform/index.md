---
title: 傅里叶变换随记
date: 2018-04-19
math: true
categories:
    - 技术经验
---
使用傅里叶变换的简易记录。  
<!--more-->  

## 原理
首先是公式  
$$ \mathcal{F}[f(t)]=\int_{-\inf}^{\inf}f(t)e^{-j\omega t}dt $$  
其中，角频率  
$$ \omega=k\omega_0=k\frac{2\pi}{T} $$  

对于离散情况，将周期 T 用 N 点来表示，在采样频率下采一个周期共 N 点，时间 t 则用 n 表示，再根据欧拉公式  
$$ e^{jx}=cosx+jsin(x) $$  
可写成  
$$ a_k = \mathcal{F}[f(n)]=\sum_{n=0}^{N-1}f(n)[cos(2\pi k\frac{n}{N})-jsin(2\pi k\frac{n}{N})] $$  
同样的，逆变换：
$$ f(n) = \frac{1}{N}\sum_{k=0}^{N-1}a_k[cos(2\pi k\frac{n}{N})+jsin(2\pi k\frac{n}{N})] $$  

## 实践
### 对称性
一般我主要是对图像或者其他实信号进行离散傅里叶变换，而因为  
$$ \begin{align*} a_k-a_{N-k} &= \sum_{n=0}^{N-1}f(n)\cdot e^{-jk\frac{2\pi}{N}n} - \sum_{n=0}^{N-1}f(n)\cdot e^{-j(N-k)\frac{2\pi}{N}n} \\ &= \sum_{n=0}^{N-1}f(n) [e^{-jk\frac{2\pi}{N}n}-1\cdot e^{jk\frac{2\pi}{N}n}] \\ &= \sum_{n=0}^{N-1}f(n)\cdot 2jsin(2\pi k\frac{n}{N}) \end{align*} $$  
当 f(n) 为实数时，值的实数为 0，说明傅里叶系数关于 N/2 对称，我们只需要计算前 N/2 个值就可以扩充为 N 个值。  

### 时域和频域
对于空域上的操作可以换算成频域上的操作，反过来也是。所以对频域上进行值加减可以等价换算成傅里叶变换后在空域全图上进行加减：  
$$ \begin{align*} Y(j\omega) &= \sum_{n=0}^{N-1}(f(n)+g(n))\cdot e^{-j\omega n}  \\ &= \sum_{n=0}^{N-1}f(n)\cdot e^{-j\omega n} + \sum_{n=0}^{N-1}g(n)\cdot e^{-j\omega n}  \\ &= F(j\omega) + G(j\omega) \end{align*}$$
乘法则是对应卷积。  

### 变换和逆变换
对于实信号而言，若只取实部做分析，在正变换和逆变换上主要是数值大小上会差 N 倍（有些库的实现会自动做这个缩放），所以无论是对时域信号还是频域信号，做正变换或逆变换得到的结果理论上都是一样的（个别库会有差别）。  
如果是连续两次相同的变换则会导致相位上偏转 90 度。
$$ \begin{align*} \mathcal{F^{-1}}\{\mathcal{F}[f(n)]\} &= \sum_{k=0}^{N-1}\sum_{n=0}^{N-1}[f(n)\cdot e^{-jk \frac{2\pi}{N}n}]e^{jk \frac{2\pi}{N}i} \\ &= \sum_{k=0}^{N-1}\sum_{n=0}^{N-1}f(n)\cdot e^{jk \frac{2\pi}{N}(i-n)} \end{align*}$$

