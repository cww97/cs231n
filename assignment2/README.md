# Assignments 2


作业说明在[这里](http://cs231n.github.io/assignments2019/assignment2/)


## Fully-connected Neural Network

今天写的时候思考了一下为什么课件里写的都是$Wx+b$的形式然而这里的code写的都是`x * w + b`，课件中的x是单张图片，这种for循环运算batch过于缓慢，故code中x是n组数据。写的时候就改成了`x*w`

backward：根据链式法则更新梯度，根据要求的shape进行顺序调整，转置，reshape等等

因为很多网络都是全连接后面立刻ReLu，所以sandwich layer就是把两层连在一起，里面调用刚刚写的两层（千万8要重复实现），嗯，这很软件工程，这种三明治层可以肆意的定义

然后用这个写好的各个layer来实现一个两层的net，作业1已经实现过一次了，这次将其解耦，原来的写了两次的全连接，这次直接调用layers里的就行了，上面刚刚说的三明治层也立刻用上

写好网络之后是对solver的介绍，把model和data（划分好train, val）扔进solver里，然后solver.train，这里把每次一边train一边val的代码也写了，就是作业1后期的很多重复劳动，很赞，偷懒是生产力发展的根本推动力。描述中说要模型练到至少50%，把`solver.py`中的例子直接复制过来就是52%了，逻辑和作业1中一模一样，效果也应当是一样的。感觉代码一下子变爽了，这个感觉以后一直能用到，tf or torch中并不会有这种solver的code，这种类似的代码似乎见过不少，api写的这么赞的是第一次。从输出中理解了epoch和Iteration的区别，前者是val loop，后者就是计算loss loop

(2020.2.8 00：32累了，剩下还有一个FullyConnectedNet各种优化方法天亮写，似乎加上batch_norm和dropout更好写，明天起来先写batch_norm然后看课程视频下一章节写了dropout再从这里继续吧，晚安~)

写BN写了一半发现不需要也行，这个依然是多层全联通网络，不过更加灵活一点，可以用一个list传入每层的节点数量（当然层数也随意）

写完还有个各种不同的优化器，直接抄公式就行

大头时间花在调参上（好像也不是很多时间），一开始自己蠢了把学习率写了个1e-2，一直爆炸，改成1e-3，一发入魂

知乎上搜的调参顺序，码住

    Andrew Ng 的个人经验和偏好是：

    第一梯队： 

    - learning rate α

    第二梯队： 

    - hidden units                  
    - mini-batch size                  
    - momentum β

    第三梯队： 
    - number of layers                  
    - learning rate decay                  
    - other optimizer hyperparameters

(2020.2.9 00:20 BN明天写吧)

JJ 说的调参顺序：
1. learning_rate
2. reg, learning_rate_decay, model_size



## Batch Normalization

[讲的比较清楚](https://zhuanlan.zhihu.com/p/34879333),在线性层和激活层直接的一层，让数据更稳定.弹幕里有说，补一些blog和paper再看视频才能看的比较明白

$$
\begin{aligned}
&\boldsymbol{\mu}=\frac{1}{N} \sum_{k=1}^{N} \boldsymbol{x}_{k}\\
&\boldsymbol{\sigma}^{2}=\frac{1}{N} \sum_{k=1}^{N}\left(\boldsymbol{x}_{k}-\boldsymbol{\mu}\right)^{2}\\
&\hat{\boldsymbol{x}}_{i}=\frac{\boldsymbol{x}_{i}-\boldsymbol{\mu}}{\sqrt{\boldsymbol{\sigma}^{2}+\boldsymbol{\epsilon}}}\\
&\boldsymbol{y}_{i}=\gamma \hat{\boldsymbol{x}}_{i}+\beta
\end{aligned}
$$

开始连的时候取20条数据练200轮，需要过拟合：排除大部分code bug ?

出现train_acc 增长，val_acc不变或者变差的时候：overfitting，增加reg


(2020.2.8 20:07  我开始了)

axis看了又忘忘了又看，[rua](https://blog.csdn.net/fangjian1204/article/details/53055219)。

BN真就抄公式呗

```python
sample_mean = np.mean(x, axis=0)
sample_var = np.var(x, axis=0)
z = (x - sample_mean) / np.sqrt(sample_var + eps)
out = gamma * z + beta
```

至于backward，我选择死亡，找个抄一下吧[死亡1](https://whu-pzhang.github.io/cs231n-assignment2/), [死亡2](https://github.com/jariasf/CS231n/blob/master/assignment2/cs231n/layers.py)

还要简化版的backward，高数没学好，["现在将这三项加在一块即可得："](https://whu-pzhang.github.io/cs231n-assignment2/#Alternative%20backward%20implement)

$$\frac{\partial L}{\partial \boldsymbol{x}_{i}}=\frac{\gamma\left(\boldsymbol{\sigma}^{2}+\epsilon\right)^{-1 / 2}}{N}\left(N \frac{\partial L}{\partial \boldsymbol{y}_{i}}-\frac{\partial L}{\partial \gamma} \cdot \hat{\boldsymbol{x}}_{j}-\frac{\partial L}{\partial \beta}\right)$$

（2020.2.9 00:28 对8起，我还没睡）

`Batch normalization and initialization`这里有确定weight_scale

weight_scales = np.logspace(-4, 0, num=20)在这段区间里均匀取20个点，然后small_test，后面还有画图的，妙啊

一次的batch越大，数据越稳定，当然训练acc越高，对于val_acc同理，不过因为验证集最终会趋于稳定，所以过大不会得到明显提升，太大反而会导致训练的很慢，所以，又学到了

另外课里面讲了各种优化方法，好像上学期课上pre刚好就是讲的这些内容，随便听听吧=- =

结论就是用Adam，另外当可以full-batch的时候用L-BFGS


## Dropout


真就抄公式呗
```python
# train
mask = (np.random.rand(*x.shape) < p)
out = x * mask

# test
out = x
```


## Convolutional Networks

[这篇](https://www.cnblogs.com/shine-lee/p/9932226.html)有卷积运算的详细解释

然后`cnn.py`写的就很起劲

然后卡在了overfit small_data上，我傻了

wq: 学习率太大了，调之，好了，牛逼



## PyTorch / TensorFlow on CIFAR-10


`if you choose to use that notebook`既然如此我就偷懒了，tf以后用到再说

pytorch 共有三个level的api

| API           | Flexibility | Convenience |
|---------------|-------------|-------------|
| Barebone      | High        | Low         |
| `nn.Module`     | High        | Medium      |
| `nn.Sequential` | Low         | High        |


似乎平常见到的最多的是 Module level的，舒服，我会了.jpg

最后一个是CIFAR-10 challeage, `model = models.resnet18(pretrained=True)`哈哈哈哈哈




