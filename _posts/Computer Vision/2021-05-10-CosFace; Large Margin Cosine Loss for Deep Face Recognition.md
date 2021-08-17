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

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="50%"></p>

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

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="60%"></p>

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

$$L_{2}$$ norm 을 그대로 loss에 사용하면 모델은 자연스럽게 특징 벡터의 $$L_{2}$$ norm 을 학습하게 된다.
이렇게 학습하면 2가지의 문제점이 생긴다고 한다.

첫번째로, 모델은 전반적인 loss를 낮추도록 학습한다.
hard sample과 easy sample이 섞여서 학습이 진행될텐데 모델이 easy sample의 $$L_{2}$$ norm을 높인다면 전반적인 loss는 쉽게 낮출 수 있다.
이런식으로 cosine metric의 부족함을 $$L_{2}$$ norm의 크기로 보완하게 되는 경우가 생긴다.

두번째로는 모델이 초기 학습에 $$L_{2}$$ norm을 줄임으로써 loss를 줄이려 하게 되는 경우이다.
$$\lVert x \rVert (\cos(\theta_{i}) - m) > \lVert x \rVert \cos(\theta_{j})$$를 목표로 학습할 때, 
아직 $$\cos(\theta_{i})-m > \cos(\theta_{j})$$가 성립하지 않는다면 $$\lVert x \rVert $$는 loss를 줄이기 위해서 작아지도록 학습되고,
이는 학습을 방해하는 요소가 된다. 어쨋건 특징을 normalize 하면 특징간의 각도만을 가지고 학습할 수 있다는 것이다.

그래서 $$s$$라는 값을 사용하는데, 이 값이 너무 작으면 또 학습이 잘 안되기 때문에 lower bound를 제시한다.
$$P_{W}$$가 expected minimum posterior probability of class center(i.e., W) 라고 할 때(이해 못해서 그대로 옮김) lower bound는 다음과 같이 주어진다.
(증명은 supplemental material에)

$$s \ge \frac{C-1}{C} \log \frac{(C-1)P_{W}}{1-P_{W}}$$



<h4>3.4. Theoretical Analysis for LMCL</h4>

$$m$$이 decision boundary에 주는 영향을 살펴본다.

$$C_{1}$$, $$C_{2}$$가 있는 이진 분류에 대해서 생각해보자. NSL의 경우 decision boundary는 $$\cos \theta_{1} - \cos \theta_{2} = 0$$으로 형성된다.
반면, LMCL은 Class 1의 경우(Class 2역시 비슷함) $$\cos \theta_{1} - \cos \theta_{2} = m$$에서 형성되고, 이는 inter-class variance를 늘리고 intra-class variance를 축소시키는 결과를 가져온다.

<p align="center"><img src="{{ img_path }}figure3.png?raw=true" width="60%"></p>

$$m$$의 범위는 margin이 아예 없는 0부터 decision boundary가 weight vector와 동일한 경우가 되도록 하는 최대 margin까지 존재할 수 있다.

$$0 \le m \le (1 - \max(W_{i}^{T} W_{j})), \mathbb{where}\ \  i,j \le n, i \ne j$$

softmax loss가 모든 weight vector들이 서로 최대의 각도를 가지도록 학습한다.
이는 hypersphere에서 weight vector들을 uniform하도록 분포시킨다는 뜻인데, 이 사실과 위의 식을 합치면 margin $$m$$을 다음과 같이 쓸 수 있다.
(증명은 supplemental material에)

$$
\begin{align}
&0 \le m \le 1 - \cos \frac{2 \pi}{C}, && (K=2) \\
&0 \le m \le \frac{C}{C-1}, && (C \le K + 1) \\
&0 \le m \ll \frac{C}{C-1}, && (C > K + 1)
\end{align}
$$

C는 training class의 수, K는 feature의 dimension이다.

2차원의 특징에 대해서 $$m$$은 최대 $$1-\cos \frac{\pi}{4}$$의 값을 가지는데 이 값은 약 0.29이다.
0부터 0.2까지 $$m$$을 늘려가며 생기는 변화를 간단하게 아래 그림으로 보여준다.

<p align="center"><img src="{{ img_path }}figure4.png?raw=true" width="80%"></p>

그런데 사실 주어진 $$m$$의 범위에서 최댓값으로 학습을 시도하면 너무 어려워진 학습 난이도 때문에 수렴이 안된다.
때문에 최적의 $$m$$ 값은 범위 내의 최댓값이 아니라 결국 실험을 통해서 얻어낸 값을 사용하는 것을 뒤에서 볼 수 있다.

---

<h3>4. Experiments</h3>

<h4>4.1. Implementation details</h4>

**Preprocessing.** MTCNN으로 얼굴 영역과 얼굴 랜드마크를 추출한다. 5개의 얼굴 포인트(양 눈, 코, 입의 양 끝)로 similarity transformation이 적용된다.
얻은 얼굴 이미지를 112x96 사이즈로 resize 해서 사용한다.

**Training.** 작은 데이터셋과 큰 데이터셋에 대해서 학습을 한다. 작은 데이터셋은 CASIA-WbFace(10,575 identities, 0.49M face images), 
큰 데이터셋(90K identities, 5M face images)은 여러 public, private 데이터셋을 묶어서 만든 것을 사용한다.
training set에는 horizontal flip이 augment로 적용된다.

+ CNN구조는 SphereFace에서 사용된 64층의 모델을 사용하며, scaling parameter $$s$$는 64로 한다.
+ SGD, batch_size=64(GPU 8대), weight_decay=0.0005.
+ 작은 데이터셋에서 초기 lr=0.1, 16K, 24K, 28K iteration에서 10으로 나눔. 30K에 학습 종료.
+ 큰 데이터셋에서 초기 lr=0.05, 80K, 140K, 200K iteration에서 10으로 나눔. 240K에 학습 종료.

**Testing.** 원본 이미지와 filp한 이미지의 특징을 concat한 것으로 최종 특징을 만든다. 
similarity score는 cosine distance를 사용, verification과 identification에서 이를 thresholding, ranking해 사용한다.


<h4>4.2. Exploratory experiments</h4>

**Effect of $$m$$.** $$m$$값은 LMCL에서 가장 중요한 역할을 한다. 
이 값에 따른 모델의 변화를 보기 위해 여러 값의 $$m$$으로 CASIA-WebFace를 학습하고 LFW와 YTF에 테스트 한다.
아래 그림에서 0.35에서 최대값이 되고 이 값을 넘어가면 증가하지 않고 되려 감소하기 때문에 이후의 실험에서 $$m$$은 0.35로 고정되어 사용된다.

<p align="center"><img src="{{ img_path }}figure5.png?raw=true" width="45%"></p>

**Effect of feature normalization.** Normalization이 효과가 있는지 알기 위해 이를 적용한 것과 안한 것으로 2번 CASIA-WebFace를 학습한다.

<p align="center"><img src="{{ img_path }}figure6.png?raw=true" width="60%"></p>

LFW, YTF, Megaface Challenge 1(MF1)에 테스트 한 결과이다. 
Rank 1은 Rank 1 정확도이고 Veri. 는 $$TAR@FAR=10^{-6}$$이다. 수치에 따르면 normalization을 하는 것이 꾸준히 좋은 성능을 내는 것을 알 수 있다.


<h4>4.3. Comparison with state-of-the-art loss functions</h4>

Loss의 차이에 따른 성능 변화를 보인다.

<p align="center"><img src="{{ img_path }}figure7.png?raw=true" width="60%"></p>


<h4>4.4. Overall benchmark comparison</h4>

<h4>4.4.1. Evaluation on LFW and YTF</h4>

LFW는 5,749 identities에 대해 13,233의 이미지가 있고, YTF는 1,595 identities에 대해 3,425의 비디오가 존재하는 데이터셋이다.
아래 표에 training data가 5M인 것은 위에 train setting 부분의 큰 데이터셋을 의미한다. 

<p align="center"><img src="{{ img_path }}figure8.png?raw=true" width="60%"></p>

<h4>4.4.2 Evlauation on MegaFace</h4>

MegaFace Challenge는 1M 개 이상의 이미지가 gallary set으로 주어지고 Facescrub과 FGNET이 probe set으로 주어진다.
본 실험에서는 Facescrub(530 identities, 106,864 face images)을 probe set으로 사용한다.

**MegaFace Challenge 1 (MF1).**

690K identities 에 대해 1M 개의 이미지가 갤러리로 주어진다. protocol은 학습할 때 이미지 수가 0.5M 미만이면 small, 이상이면 Large로 분류된다.

<p align="center"><img src="{{ img_path }}figure9.png?raw=true" width="60%"></p>

**MegaFace Challenge 2 (MF2).**

MF2는 Megaface에서 제공하는 데이터로 학습을 진행해야 한다. 672K identities, 4.7M face images를 가지기 때문에 protocol은 Large로 분류된다.

<p align="center"><img src="{{ img_path }}figure10.png?raw=true" width="60%"></p>


---

<h3>5. Conclusion</h3>

pass

---

<h3>Supplementary material</h3>

굳이 쓸필요 없을듯