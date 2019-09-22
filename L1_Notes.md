# Word2vec

* Each work in a fixed vocabulary is represented by a vector 

* The centre word and context words are represented as 'c' and 'o' in the following

* Use the *the similarity of the word vectors* for c and o to *calculate the probability* of o given c (or vice versa)

* Adjust the vector to maximize this probability

  ![1569041747891](C:\Users\gao_x\AppData\Roaming\Typora\typora-user-images\1569041747891.png)


$$
\mathsf{Likelibood}=L(\theta)=\prod_{t=1}^{T}\prod_{-m \leq j \leq m, j \neq 0} P(w_{t+j} |w_t; \theta)
$$

where $\theta$ is all variables to be optimized and $m$ is the fixed size of the window.

The cost function $J(\theta)$ is the average negative log likelihood:
$$
J(\theta)=-\frac{1}{T}\log L(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\sum_{-m\le j \le m}\log P(w_{t+j}|w_t;\theta)
$$
For a centre word *c* and a context word *o*, the probability is as follows:
$$
P(o|c)=\frac{\exp (u_o^Tv_c)}{\sum_{w \in V}\exp(u_w^Tv_c)}
$$
where $v_c$ and $u_w$ are two vectors for the centre word and the context word. $V$ is the entire vocabulary.

The above expression shows the similarity of $o$ and $c$ normalized over the entire vocabulary, which is an example of the softmax function $\mathbb{R}^n \rightarrow \mathbb{R}^n$ 
$$
\mathsf{softmax}(x_i)=\frac{\exp(x_i)}{\sum_{j-1}^n \exp(x_j)}=p_i
$$
The softmax function maps arbitrary values $x_i$ to a probability distribution $p_i$:

* "max" because amplifies probability of largest $x_i$
* "soft" because still assign some probability to smaller $x_i$

Recall that $\theta$ represents all model parameters in one long vector. In our case with *d*-dimensional vectors and *V*-many words. Consider that every word has two vectors, therefor $\theta$ is a 2V x d matrix. For instance:
$$
\theta = \left[\begin{array}{cols} v_{aardvark}\\v_a\\.\\.\\.\\v_{zebra}\\u_{aardvark}\\u_a\\.\\.\\.\\u_{zebra}\end{array} \right] \in \mathbb{R}^{2dV}
$$
