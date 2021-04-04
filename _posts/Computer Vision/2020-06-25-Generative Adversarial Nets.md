---
layout: post
title: Generative Adversarial Nets
tag: [ComputerVision, Unclassified]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" %}

GAN을 공부하게 되면서 가장 처음 모델을 제안했던 논문으로 돌아가 분석을 해보기로 했다.

<h3>1. Introduction</h3>

<abbr title="Generative Adversarial Network">GAN</abbr>은 생성적 적대 신경망 으로 해석할 수 있다.
Generative 모델을 만드는데 적대적으로 학습시키겠다는 뜻이다. Discriminator과 Generator를 적대적으로 학습시킬 것이다.
Generator를 위조지폐범이라 하고 Discriminator를 경찰에 비유한다면 Generator는 경찰에 발각되지 않는 위조지폐를 만들어야 하고,
경찰은 위조지폐를 잘 구분해내야 할 것이다. 경쟁하기 때문에 각자의 능력을 향상시켜야 한다.

Generator와 Discriminator 둘 다 Multi Layer Perceptron으로 구성된다. 
Generator는 random noise에서 sample을 생성하고, Discriminator는 dataset의 실제 데이터와 Generator의 sample을 받아서 분류 한다.

---

<h3>2. Related Work</h3>

pass

---

<h3>3. Adversarial nets</h3>

$$ p_{data} $$ : 실제 데이터의 분포

$$ p_{g} $$ : generator's distribution이라 나와있는데, generator가 input(noise)을 받아서 만들어내는 데이터 분포 정도로 이해한다. 

$$ p_{z}(z) $$ : input noise $$ z $$ 에 대한 prior

$$ G(z;\theta _{g}) $$ : input noise $$ z $$ 에서의 data space로의 mapping 표현. parameter $$ \theta_{g} $$를 가지는 multilayer perceptron이다.

$$ D(x;\theta_{d}) $$ : parameter $$ \theta_{d} $$를 가지는 multilater perceptron. $$ x $$를 입력 받아서 single scalar 값을 반환한다. $$ D(x) $$는 $$ x $$가 data에서 비롯된 것인지, $$ p_{g} $$ 에서 비롯된 것인지의 확률 값이 된다.

Adversarial nets은 D와 G의 value function $$ V(G, D) $$에 대한 minmax game이다.

$$ \min_{G} \max_{D} V(D,G)=\mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1-D(G(z)))]\tag{1}$$

G는 V 값을 낮게 하는 방향으로, D는 V 값을 높게 하는 방향으로 학습하겠다는 뜻이다. 
식을 잘 쳐다보고 있으면 기댓값은 D가 분류를 잘할수록 0으로 수렴하고, 못할수록 $$ -\infty $$로 발산하는 것을 알 수 있다.
G는 D를 속이고, D는 G에게 속지 않으려 한다는 소개에서의 컨셉과 일치하는 식이다.

그림을 통해 표현한다면 다음과 같이 된다.

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="100%"></p>
*discriminative distribution(D, blue, dashed line), data distribution(black, dotted line), generative distribution(green, solid line)*

(a) 에서 실 데이터의 분포와 생성 데이터의 분포가 다르게 생겼고, 분류 모델은 어느정도 구별은 하지만 확률값이 출렁대는 것이 보인다.

(b) D를 몇 회 학습시킨다. 분류 모델이 수렴하면 파란 선이 $$ \frac{p_{data}(x)} {p_{data}(x)+p_{g}(x)} $$을 따른다.

(c) D의 출력 값에 따라서 G가 그럴듯한 데이터를 내놓도록 분포를 조정한다.

(d) 이 과정을 몇 번 거치다 보면 더 개선시킬 여지가 없는 $$ p_{g}=p_{data} $$에 도달하게 된다(G와 D의 capacity가 충분하다면). 이 때 $$ D(x)=\frac{1}{2} $$이다.

실제에 적용할 때 equation 1은 초기 학습시에 좋지 않을 때가 있다. 
G가 아직 제대로 된 값을 못내놓는다면 D가 완벽한 분류를 해내서 $$ \log (1-D(G(z))) $$가 saturate되는데, equation 1은 이 경우 G에 gradient를 전해줄 수 없게된다. 
실제 학습 시킬 때는 $$ \log D(G(z)) $$를 maximize 하도록 하는 것이 같은 뜻이지만 D가 강한 경우에도 더 나은 gradient를 줄 수 있다.

$$
\begin{align*}
\max_{D} V(D,G)& =\mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1-D(G(z)))] \\
\max_{G} V(D,G)& =\mathbb{E}_{z~p_{z}(z)} [\log D(G(z))]
\end{align*}
$$

---

<h3>4. Theoretical Results</h3>

GAN을 수렴시키기 위한 알고리즘은 다음과 같이 제안된다. $$ k $$번 D를 학습시킨 다음 G를 학습시킨다. 
stochastic gradient descent가 사용되며 loss function은 위에 쓰인 것을 사용한다.

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="100%"></p>

<h4> 4.1 Global Optimality of $ p_{g}=p_{data} $ </h4>

**Proposition 1.** G가 고정되었을 때, the optimal discriminator D는

$$ D^{*}_{G}(x)=\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)} \tag{2}$$

*proof.* D의 목적은 V를 최대화 하는 것이다. V는


$$ 
\begin{align*} 
V(G,D) & = \int_{x} p_{data}(x) \log (D(x))dx + \int_{z} p_{z}(z) \log (1-D(g(z)))dz
\\     & = \int_{x} p_{data}(x) \log (D(x))dx + p_{g}(x) \log (1-D(x))dx \tag{3}
\end{align*}
$$

(1)번 식의 Expectation울 풀어 쓰면 3번이다. 여기서 

$$ \mathbb{E}_{z \sim p_{z}}[\log (1-D(G(z)))] = \mathbb{E}_{x \sim p_{g}}[\log (1-D(x)))] $$

라는 사실에서 식이 전개되는데, 이 등식이 어떻게 성립하는지는 아직 이해하지 못했다.

(3)번 식에서 $$D$$가 $$V(G,D)$$를 최대화 한다는 뜻은 $$p_{data}(x) \log (D(x))dx + p_{g}(x) \log (1-D(x))$$가 모든 $$x$$에 대해 최대값을 가져야 한다. 
식을 간단하게 써보자.

$$ f(y) = a \log y + b \log (1-y) $$

얘를 미분해주면

$$ 
\begin{align*} 
f'(y) &= \frac{a}{y} - \frac{b}{1-y} \\
f''(y) &= -\frac{a}{y^{2}} - \frac{b}{(1-y)^{2}}
\end{align*}
$$

$$y= \frac{a}{a+b}$$에서 $$a+b \ne 0$$가 아니라면 $$f(y)$$가 최소 혹은 최대값을 가지는 것을 도함수로부터 알 수 있다.
이계도함수에 $$\frac{a}{a+b}$$를 넣어보면 음수값이 나오기 때문에 최대값을 가진다는 것을 알 수 있다. 
이는 a와 b를 (0, 1)의 범위 내에서만 생각하면 되기 때문인데, a와 b가 $$p_{data}$$와 $$p_{g}$$인 것을 생각하면, 그 밖의 범위에서는 정의될 필요가 없다는 것을 알 수 있다.

위에서 D가 최적인 상태에서의 값을 구했다. G는 D가 최적인 상태라고 가정한 상태에서 학습을 하므로 criterion을 다음과 같이 쓸 수 있다.

$$
\begin{align*}
C(G) &= \max_{D} V(G,D)
\\   &= \mathbb{E}_{x \sim p_{data}} [\log D_{G}^{*}(x)] + \mathbb{E}_{z \sim p_{z}} [\log (1-D_{G}^{*}(G(z)))]
\\   &= \mathbb{E}_{x \sim p_{data}} [\log D_{G}^{*}(x)] + \mathbb{E}_{x \sim p_{g}} [\log (1-D_{G}^{*}(x))]
\\   &= \mathbb{E}_{x \sim p_{data}} \left[ \log \frac{p_{data}(x)}{p_{data}(x) + p_{g}(x)} \right] + \mathbb{E}_{x \sim p_{g}} \left[ \log \frac{p_{g}(x)}{p_{data}(x) + p_{g}(x)} \right] \tag{4}
\end{align*}
$$

**Theorem 1.** virtual training criterion C(G)의 global minimum은 $$ p_{g} = p_{data} $$의 필요충분조건이며, 이때 C(G)의 값은 $$ -\log 4 $$이다.

*proof.* (2)번 식에 따라서 $$ p_{g}=p_{data} $$일 때 $$D_{ㅎ}^{*}=\frac{1}{2}$$이다. 그렇다면 (4)번 식에 따라서 $$ C(G)= -\log 4$$이다. 
필요충분조건이기 때문에 이제 $$C(G)= -\log 4$$일 때 $$p_{g} = p_{data}$$임도 보여야 한다.

$$ V(G,D) = \int_{x} p_{data}(x) \log \left( \frac{p_{data}(x)}{p_{data}(x) + p_{g}(x)} \right) dx + \int_{x} p_{g}(x) \log \left( \frac{p_{data}(x)}{p_{g}(x) + p_{g}(x)} \right) dx $$

이 식을 Kullback-Leibler divergence로 쓸 수도 있는데, 이것은 대칭성을 가지지 않는다. 대칭성을 가지는 Jensen-Shannon divergence로 고쳐보자.

$$
\begin{align*}
C(G) &= \int_{x} p_{data}(x) \log \left( \frac{p_{data}(x)}{p_{data}(x) + p_{g}(x)} \right) dx + \int_{x} p_{g}(x) \log \left( \frac{p_{g}(x)}{p_{data}(x) + p_{g}(x)} \right) dx
\\   &= \int_{x} p_{data}(x) \log \left( \frac{p_{data}(x)}{p_{data}(x) + p_{g}(x)} \right) dx + \int_{x} p_{g}(x) \log \left( \frac{p_{g}(x)}{p_{data}(x) + p_{g}(x)} \right) dx 
\\   &\quad + \int_{x} (\log 2 - \log 2)p_{data}(x) + \int_{x} (\log 2 - \log 2)p_{g}(x)
\\   &= \int_{x} p_{data}(x) \left( \log 2 + \log \left( \frac{p_{data}(x)}{p_{data}(x) + p_{g}(x)} \right) \right) dx + \int_{x} p_{g}(x) \left( \log 2 + \log \left( \frac{p_{g}(x)}{p_{data}(x) + p_{g}(x)} \right) \right) dx
\\   &\quad - \log 2 \int_{x} p_{g}(x) + p_{data}(x) dx
\\   &= -\log 4 + \int_{x} p_{data}(x) \log \left( \frac{p_{data}(x)}{(p_{data}(x) + p_{g}(x)) / 2} \right) dx + \int_{x} p_{g}(x) \log \left( \frac{p_{g}(x)}{(p_{data}(x) + p_{g}(x)) / 2} \right) dx 
\\   &= -\log (4) +KL \left( p_{data} \parallel \frac{p_{data} + p_{g}}{2} \right) + KL \left( p_{g} \parallel \frac{p_{data}+p_{g}}{2} \right) \tag{5}
\end{align*}
$$

$$\int_{x} p_{data}(x) dx$$와 $$\int_{x} p_{g}(x) dx$$는 1이다. (5)번 식의 KL을 JSD로 바꿔보자.

$$ C(G) = -\log (4) + 2 \cdot JSD (p_{data} \parallel p_{g}) \tag{6} $$

JSD는 두 분포가 같을 때 0을 가지고, 이외에는 양수를 가진다. 따라서 $$C(G)$$의 global optimum은 $$p_{g}=p_{data}$$일 때 $$C^{*}=-\log (4)$$이다.

<h4>4.2 Convergence of Algorithm 1</h4>

**Proposition 2.** G와 D가 충분한 capacity를 가지고 있다고 한다. Algorithm 1의 각 step마다 discriminator는 주어진 G에 의해 optimum에 수렴하고, G는 criterion을 improve하기 위해 값이 조정된다. 이에 $$p_{g}$$는 $$p_{data}$$에 수렴한다.

pass

---

<h3>5. Experiments</h3>

pass