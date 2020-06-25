---
layout: post
title: Generative Adversarial Nets
tag: [papers]
---

GAN을 공부하게 되면서 가장 처음 모델을 제안했던 논문으로 돌아가 분석을 해보기로 했다.

<h3>1. Introduction</h3>

<abbr title="Generative Adversarial Network">GAN</abbr>은 적대적 생성 신경망(혹은 생성적 적대 신경망? 둘다 검색해도 GAN 정보가 나온다) 정도로 해석할 수 있다. 
무엇이 적대적이고, 무엇이 생성적인가? 저자는 D(Descriminative model)와 G(Generative model) 2개의 모델을 소개하며 이 둘을 동시에 적대적으로 학습시키는 것을 제안한다. 그리고 여기서 G가 최종적으로 얻게되는, 데이터를 생성하는 모델이다.

D와 G의 역할은 간단하다. D는 입력받은 데이터가 dataset(실제 데이터)에서 왔는지 G가 생성해낸 가짜 데이터인지를 구분한다. G는 D가 분류가 실패하도록 가짜 데이터를 찍어내는 역할을 한다. 저자는 이를 경찰과 위조지폐범에 비유했는데, 경찰은 위조지폐를 잡아내기 위해 노력하고, 위조지폐범은 안들키고 쓸 수 있는 지폐를 만들기 위해 개선을 하는 것이다. 

G가 dataset의 distribution을 정확히 잡아내서 D에 넘겨준다면, 즉 G가 실제 데이터와 똑같은 가짜 데이터를 만들어낸다면, D가 데이터 분포 내에서 진짜와 가짜를 분류할 확률은 언제나 50%일 것이다.

<h3>2. Related Work</h3>

여긴 딱히 주의깊게 안읽었다.

<h3>3. Adversarial nets</h3>

저자가 사용한 표현들은 다음과 같다. 

$$ p_{data} $$ : 실제 데이터의 분포

$$ p_{g} $$ : 실제 데이터의 분포를 학습한 generator의 data distribution이 된다. 처음에는 noise에 대한 표현으로 $$ p_{z}(z) $$라고 정의한다. 

$$ G(z;\theta _{g}) $$ : input noise $$ z $$ 에서의 data space로의 mapping 표현. parameter $$ \theta_{g} $$를 가지는 multilayer perceptron이다.

$$ D(x;\theta_{d}) $$ : parameter $$ \theta_{d} $$를 가지는 multilater perceptron. $$ x $$를 입력 받아서 single scalar 값을 반환한다. $$ D(x) $$는 $$ x $$가 data에서 비롯된 것인지, $$ p_{g} $$ 에서 비롯된 것인지의 확률 값이 된다.

Adversarial nets은 D와 G라는 two-player의 value function $$ V(G, D) $$에 대한 minmax game이라고 표현된다.

$$ \min_{G} \max_{D} V(D,G)=\mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1-D(G(z)))]\tag{1}$$

G는 V 값을 낮게 하는 방향으로, D는 V 값을 높게 하는 방향으로 parameter들을 학습하겠다는 뜻이다. 

$$ \mathbb{E}$$는 기대 값으로 사건의 값과 확률을 곱해서 다 더한 것이다. $$ x \sim p_{data}(x) $$는 data의 확률분포에 대한 $$ x $$ 값들에 대하여 정도로 해석할 수 있겠다. $$ D(x) $$가 확률 값으로 0~1의 값을 가지므로 $$ \log D(x) $$ 는 $$ -\infty \sim 0 $$의 값을 가질 수 있게 된다. 뒤의 항도 비슷한 내용이니 D가 완벽한 분류를 해낸다면 V값은 0이 될 것이고, 분류가 완전 엉망이 된다면 음의 무한대를 향해서 갈 것이다.

그림을 통해 표현한다면 다음과 같이 된다.

![figure1](https://github.com/illian01/illian01.github.io/blob/master/assets/Generative%20Adversarial%20Nets_figure1.PNG?raw=true)
*discriminative distribution(D, blue, dashed line), data distribution(black, dotted line), generative distribution(green, solid line)*

(a) 에서 실 데이터의 분포와 생성 데이터의 분포가 다르게 생겼고, 분류 모델은 어느정도 구별은 하지만 확률값이 출렁대는 것이 보인다.

(b) inner loop를 통해 분류 모델이 수렴하면 파란 선이 $$ \frac{p_{data}(x)} {p_{data}(x)+p_{g}(x)} $$을 따른다.

(c) D의 값에 따라서 G가 그럴듯한 데이터를 내놓도록 분포를 조정한다.

(d) 이 과정을 몇 번 거치다 보면 더 개선시킬 여지가 없는 $$ p_{g}=p_{data} $$에 도달하게 된다(G와 D의 capacity가 충분하다면). 이 때 $$ D(x)=\frac{1}{2} $$이다.

실제에 적용할 때 equation 1은 초기 학습시에 좋지 않을 때가 있다. G가 아직 제대로 된 값을 못내놓는다면 D가 완벽한 분류를 해내서 $$ \log (1-D(G(z))) $$가 saturate된다. 따라서 $$ \log D(G(z)) $$를 maximize 하도록 학습 시키는 것이 같은 뜻이지만 더 나은 gradient를 줄 수 있다고 한다.


<h3>4. Theoretical Results</h3>

GAN을 수렴시키기 위한 알고리즘은 다음과 같이 제안된다.

![figure1](https://github.com/illian01/illian01.github.io/blob/master/assets/Generative%20Adversarial%20Nets_figure2.PNG?raw=true)
 
<h4> 4.1 Global Optimality of $ p_{g}=p_{data} $ </h4>

**Proposition 1.** For G fixed, the optimal discriminator D is

$$ D^{*}_{G}(x)=\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)} \tag{2}$$

*proof.* D의 목적은 V를 최대화 하는 것이다. V는


$$ 
\begin{align*} 
V(G,D) & = \int_{x} p_{data}(x) \log (D(x))dx + \int_{z} p_{z}(z) \log (1-D(g(z)))dz
\\     & = \int_{x} p_{data}(x) \log (D(x))dx + p_{g}(x) \log (1-D(x))dx \tag{3}
\end{align*}
$$

(1)번 식을 좀 더 풀어서 쓴 것이 3번이다. dataset에서 온 $$x$$와 noise $$ z $$ 에서 온 $$ G(z) $$의 기댓값을 더한 것이 value function G이다. 근데 이게 사실 D의 입장에서는 입력이 모두 $$ x $$로 들어오고 분류를 하는건 자신의 몫이니 x가 dataset으로 비롯된다고 판단되면 $$p_{data}(x)$$를 곱하는 것이고, 가짜로 판단된다면 $$p_{g}(x)$$를 곱해서 더하면 된다. 그렇게 하면 위의 기댓값 식과 같은 의미로 식을 쓸 수 있다.

적분 내부 식을 다시 써보자. 양의 실수 $$a$$, $$b$$에 대해서 함수 $$a \log (y) + b \log (1-y)$$로 쓸 수 있다.
y는 (0,1)에서 값을 가지고, 미분해서 0이되는 지점을 찾으면 $$\frac{a}{a+b}$$ 에서 최댓값을 가진다는 것을 알 수 있다. Descriminator 는 $$Supp(p_{data}) \cup Supp(p_{g})$$ 밖에서 정의 될 필요가 없다고 하는데, Support가

$$supp\ f=cl\{x\ \in X: f(x) \ne 0 \}$$

인 집합이라고 한다. data의 분포, generator의 분포 밖에서 정의 될 필요가 없다고 하는 것은 참으로 맞는 말이다. 여기서 증명을 마친다.

updating...