# Equation Index

Use this page for rapid whiteboard review.

## ML


$$
\hat\theta=\arg\min_\theta \frac1N\sum_i\ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta)
$$


## Softmax Cross Entropy


$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},\qquad
L=-\log p_y
$$


## Attention


$$
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$


## RoPE relative-position identity


$$
(R_mq)^\top(R_nk)=q^\top R_{n-m}k
$$


## Autoregressive LM


$$
p(x_{1:T})=\prod_t p(x_t|x_{<t})
$$


## LoRA


$$
W'=W+\frac{\alpha}{r}BA
$$


## CLIP / InfoNCE


$$
L_i=-\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}
$$


## DDPM


$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$


## Bellman


$$
V^\pi(s)=\mathbb E[r+\gamma V^\pi(s')]
$$


## Policy Gradient


$$
\nabla_\theta J=
\mathbb E[\nabla_\theta\log\pi_\theta(a|s)A(s,a)]
$$


## PPO


$$
L^{CLIP}
=
\mathbb E[
\min(r_tA_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
]
$$


## World Model


$$
p_\theta(z_{t+1}|z_t,a_t)
$$


## RAG


$$
D_k(q)=\mathrm{TopK}_d\,s(f_q(q),f_d(d))
$$


