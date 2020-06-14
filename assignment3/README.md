# cs231n Assignment3


http://cs231n.github.io/assignments2019/assignment3/


## RNN

可能是由于自己机器的一些奇怪的设定，太难了

```python
def image_from_url(url):
    #########################  interesting  ##############################
    proxy = urllib.request.ProxyHandler({'http': '127.0.0.1:12333'})
    opener = urllib.request.build_opener(proxy,urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    ######################################################################
    f = urllib.request.urlopen(url)
    ...
```

$$
\begin{aligned}
&h_{t}=f_{W}\left(h_{t-1}, x_{t}\right)\\
&11\\
&\begin{array}{l}
{h_{t}=\tanh \left(W_{h h} h_{t-1}+W_{x h} x_{t}\right)} \\
{y_{t}=W_{h y} h_{t}}
\end{array}
\end{aligned}
$$

主要工作再`rnn_layer`和`rnn`，先写没一层的code然后combine，依然是算梯度比较痛苦。后面的基本上把每个paramenter的shape摸清楚就是很简单（....吧）的组装。


## LSTM

$$
\begin{align*}
i = \sigma(a_i) \hspace{2pc}
f = \sigma(a_f) \hspace{2pc}
o = \sigma(a_o) \hspace{2pc}
g = \tanh(a_g)
\end{align*}
$$

Finally we compute the next cell state $c_t$ and next hidden state $h_t$ as

$$
c_{t} = f\odot c_{t-1} + i\odot g \hspace{4pc}
h_t = o\odot\tanh(c_t)
$$

where $\odot$ is the elementwise product of vectors.

JJ讲的很详细很赞，感觉把公式里每个符号摸清楚了就很容易写了，forward水到渠成，backward各凭本事


## Network Visualization



## Style Transfer



## GAN



