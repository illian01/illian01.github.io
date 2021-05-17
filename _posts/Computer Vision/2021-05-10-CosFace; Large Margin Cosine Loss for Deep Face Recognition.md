---
layout: post
title: CosFace; Large Margin Cosine Loss for Deep Face Recognition
tag: [ComputerVision, FaceRecognition]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" %}

<h3>CVPR, 2018</h3>

---

<h3>1. Introduction</h3>

이론적인 배경은 A-Softmax와 같다. 좀 더 다른 것은 softmax에서 angular margin 대신 cosine margin을 부여하고 weight vector와 더불어 feature도 normalize한다.

<p align="center"><img src="/assets/Computer Vision/CosFace; Large Margin Cosine Loss for Deep Face Recognition/figure1.png" width="50%"></p>

논문의 요약은 다음과 같다.

1. intra-class variance를 최소화하고 inter-class variance를 최대화하는 LMCL이라는 loss function을 소개한다.
2. 이론적인 분석을 한다.
3. LFW, YTF, Megaface에 대해서 sota를 달성함을 보인다.

---

<h3>2. Related work</h3>

pass

---

<h3>3. Proposed approach</h3>

<h4>3.1. Large margin cosine loss</h4>

입력 특징 벡터 $$x_{i}$$와 상응하는 레이블 $$y_{i}$$에 대해서 softmax는 다음과 같이 쓰인다.

$$L_{s} = \frac{1}{N} \sum^{N}_{i=1}-\log p_{i} = \frac{1}{N} \sum^{N}_{i=1}-\log \frac{e^{f_{y_{i}}}}{\Sigma^{C}_{j=1}e^{f_{j}}}$$

$$p_{i}$$는 $$x_{i}$$의 posterior가 되겠다. $$C$$가 class의 수이고 $$f_{j}$$는 각 클래스의 fully connected 연산이다.
weight vector $$W_{j}$$를 가진다고 하고 bias는 무시하면 다음과 같이 된다.

$$f_{j} = W^{T}_{j}x = \lVert W_{j} \rVert \lVert x \rVert \cos \theta_{j}$$

그러면 여기서 $$\theta_{j}$$는 $$x$$와 $$W_{j}$$의 사이각이다. 
SphereFace와 같이 학습 단계에서 매 단계마다 $$\lVert W_{j} \rVert=1$$로 정규화하고, $$\lVert x \rVert=s$$로 상수처리한다.
이렇게 나온 식을 Normalized version of Softmax Loss라 해서 NSL이라 표기한다.

$$L_{ns} = \frac{1}{N} \sum_{i}-\log \frac{e^{s\cos(\theta_{y_{i}, i})}}{e^{s\cos(\theta_{j, i})}}$$

여기에 cosine margin을 추가한다. A-Softmax에서는 $$\theta$$에 $$m$$을 곱했는데, 여기서는 $$\cos(\theta)$$에 $$m$$을 빼서 margin을 형성한다.
이것은 Large Margin Cosine Loss라 해서 LMCL로 표기한다.

$$L_{lmc} = \frac{1}{N}\sum_{i}-\log \frac{e^{s(\cos(\theta_{y_{i}, j})-m)}}{e^{s(\cos(\theta_{y_{i}, j})-m)} + \Sigma_{j\ne y_{i}}e^{s\cos(\theta_{j, i})}}$$

<h4>3.2. Comparison on different loss functions</h4>

<p align="center"><img src="/assets/Computer Vision/CosFace; Large Margin Cosine Loss for Deep Face Recognition/figure2.png" width="60%"></p>

$$C_{1}$$과 $$C_{2}$$ 2개의 클래스가 있는 이진 분류로 가정한다. 각 클래스의 weight vector는 $$W_{1}, W_{2}$$이다.

**Softmax.** Softmax의 decision boundary는 다음과 같다.

$$\lVert W_{1} \rVert \cos(\theta_{1}) = \lVert W_{2} \rVert \cos(\theta_{2})$$

cosine 공간 안에서 weight의 크기에 따라서 decision boundary가 왔다갔다하는데, 왜 저자가 decision area가 겹친다고 표현했는지는 모르겠다.

**NSL.** weight의 크기를 1로 정규화해서 decision boundary는 $$\cos(\theta)$$에 의해서만 결정된다.

$$\cos(\theta_{1}) = \cos(\theta_{2})$$

**A-Softmax.** A-Softmax는 angular margin $$m$$을 추가한다.

$$
\begin{align*}
C_{1} &: \cos(m\theta_{1}) \ge \cos(\theta_{2})\\
C_{2} &: \cos(m\theta_{2}) \ge \cos(\theta_{1})
\end{align*}
$$

$$C_{1}$$으로 분류하기 위해서 $$\theta_{1} \le \frac{\theta_{2}}{m}$$이 성립해야 한다는 뜻이다.
여기에 문제점은 margin이 모든 $$\theta$$에 걸쳐서 일정하지 않다는 것이다. $$\theta$$가 0으로 수렴할수록 margin은 사라진다.
만일 $$C_{1}, C_{2}$$두 클래스가 너무 유사해서 $$W_{1}, W_{2}$$가 비슷한 방향으로 간다면 margin은 크게 형성 될 수 없고, 학습이 쉽지 않을 것이다.
또한 $$\theta$$의 범위 때문에 도입된 $$\psi$$함수가 복잡한 것도 문제점이라면 문제점이라고 할 수 있겠다.

**LMCL.** cosine 공간에서 $$m$$을 패널티로 줌으로 $$\sqrt{2}m$$이라는 마진이 형성된다. 각에 상관없이 코사인 값에 패널티를 주는 것이라 훨씬 강인하다.

$$
\begin{align*}
C_{1} &: \cos(\theta_{1}) \ge \cos(\theta_{2}) + m\\
C_{2} &: \cos(\theta_{2}) \ge \cos(\theta_{1}) + m
\end{align*}
$$


<h4>3.3. Normalization of features</h4>